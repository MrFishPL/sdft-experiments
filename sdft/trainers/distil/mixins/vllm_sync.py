from sdft.trainers.distil._imports import *


class VLLMSyncMixin:
    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Select which model to sync to vLLM: teacher (ref_model) or student (model)
        # When generate_from_teacher=True, sync the teacher model since vLLM was initialized with teacher weights
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            if self.generate_from_teacher:
                raise ValueError("PEFT model handling only applies when syncing student model (teacher is typically not PEFT)")
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model_to_sync)
            else:
                for name, param in model_to_sync.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

