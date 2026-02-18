from sdft.trainers.distil._imports import *


class LossMixin:
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        
        # Create a separate mask for loss computation that skips the first N tokens
        # Note: completion_mask is used for both attention (forward pass) and loss computation
        # We need to keep the original for attention, but create a modified one for loss
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            # Create a mask that is 0 for the first num_loss_tokens_to_skip tokens and 1 elsewhere
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            # Apply the skip mask (only mask tokens that were originally unmasked)
            loss_completion_mask = completion_mask * skip_mask
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        if self.alpha == 0: #Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1: #Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]),
                dim=0,
            )

            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)

        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        # `current_gradient_accumulation_steps` may be unset before the first training step
        # (e.g., when running an initial baseline eval before `train()`).
        grad_accum_steps = getattr(self, "current_gradient_accumulation_steps", 1)
        if grad_accum_steps is None or grad_accum_steps <= 0:
            grad_accum_steps = 1
        loss = loss / grad_accum_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())
        
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None
