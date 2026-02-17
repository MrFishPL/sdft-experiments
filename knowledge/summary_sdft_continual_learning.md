# Self-Distillation Enables Continual Learning (arXiv: 2601.19897)

Paper URL: https://arxiv.org/pdf/2601.19897  
Source read from: `~/.cache/sdft/knowledge/2601.19897/arxiv.tex`

## TL;DR

The paper introduces **Self-Distillation Fine-Tuning (SDFT)**: an on-policy method for learning from demonstrations without an explicit reward model.  
The same model is used in two roles:

- student: conditioned on prompt only
- teacher: conditioned on prompt + demonstration

Training minimizes KL between student and teacher on **student-generated trajectories**, which reduces forgetting compared to SFT while improving new-task performance.

## Method Summary

- Input per example: prompt `x`, demonstration `c`
- Student policy: `pi_theta(. | x)`
- Teacher policy: `pi_phi(. | x, c)` (paper default: EMA teacher)
- Rollout: sample completion from student
- Loss: distill teacher distribution onto student along sampled trajectory
- Update:
  - optimize student with KL-based token loss
  - update teacher via EMA: `phi <- alpha * theta + (1 - alpha) * phi`

Teacher prompt template in paper:

```text
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

The paper also frames this as IRL: in-context conditioning provides an implicit reward-like signal.

## Main Experimental Claims

- Across skill learning (science QA, tool use, medical) and knowledge acquisition, SDFT beats SFT on the new task while better preserving prior capabilities.
- Sequential training across 3 tasks shows SDFT keeps prior skills much better than SFT.
- On the knowledge benchmark, SDFT improves strict and OOD accuracy over SFT.
- For reasoning models trained with answer-only data, SDFT improves accuracy while preserving longer reasoning traces better than SFT.
- Ablations show:
  - on-policy distillation matters (offline distillation underperforms)
  - EMA teacher is more stable than frozen teacher or raw student teacher
  - masking first few completion tokens helps suppress teacher-context artifacts

## Paper-to-Repo Alignment

This repo implements the paper’s core mechanics directly.

1. Demonstration-conditioned teacher prompt
- `sdft/data/tooluse.py` builds `teacher_prompt` with the same phrasing used in the paper.

2. On-policy rollout with student trajectory reuse
- `sdft/trainers/distil/mixins/generation.py` generates completions from current policy and computes teacher/student log-probs on those sampled completions.
- `sdft/trainers/distil/mixins/sampling.py` and generation buffering implement grouped generation per optimization cycle.

3. KL-based distillation loss over full token distributions
- `sdft/trainers/distil/mixins/loss.py` computes full-distribution KL-like token losses from `all_logps` and `teacher_all_logps`.

4. EMA-style teacher synchronization
- `sdft/trainers/distil/callbacks.py` applies `ref = alpha * model + (1 - alpha) * ref` on sync steps.
- `scripts/train.py` enables sync each step (`sync_ref_model=True`, `ref_model_sync_steps=1`).

5. vLLM mismatch correction (paper appendix mentions this)
- `generation.py` + `loss.py` implement importance-sampling correction (`vllm_importance_sampling_correction`).

6. Loss masking of initial tokens (artifact mitigation)
- `num_loss_tokens_to_skip` is implemented in `loss.py` and enabled in `scripts/train.py` with value `3`.

## Notes / Reproduction Checks

- The paper appendix says they use one trajectory per prompt; this codebase defaults `num_generations=8` unless overridden in config.
- Divergence mode is configurable with `alpha` in `DistilConfig`. Verify the exact KL direction you want when reproducing paper numbers.
- The current repo is focused on ToolUse training code path and core trainer internals; it does not include full benchmark/eval harness scripts from the paper.

## Practical Takeaways for This Repo

- The implementation is not just inspired by SDFT; it is a direct operationalization of the paper’s main algorithmic loop.
- If your goal is strict paper-faithful replication, first pin:
  - `num_generations`
  - KL direction (`alpha`)
  - EMA sync hyperparameters (`ref_model_mixup_alpha`, `ref_model_sync_steps`)
  - max completion length per experiment type
