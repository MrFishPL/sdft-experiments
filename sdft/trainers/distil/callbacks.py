from sdft.trainers.distil._imports import *
class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.
    
    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size), making it feasible to sync
    large models with DeepSpeed ZeRO-3.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.
        
        This is O(1) in memory overhead instead of O(N) where N is model size.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        
        if is_zero3:
            import deepspeed
            
            # Iterate through parameters one at a time
            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                # Gather only this pair of parameters
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            # Non-ZeRO-3: just iterate normally
            for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
                MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
