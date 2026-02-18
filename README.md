# Example run (NVIDIA Spark, 140GB VRAM)

```bash
NUM_GPUS=1
CUDA_DEVICES=0
DEVICE_BS=8
ACCUM_STEPS=4
NUM_GENERATIONS=4
EFFECTIVE_BS=$((DEVICE_BS * ACCUM_STEPS * NUM_GPUS))
echo "effective_batch_size=${EFFECTIVE_BS} (= ${DEVICE_BS} * ${ACCUM_STEPS} * ${NUM_GPUS})"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/home/$USER/sdft-experiments/runs/tooluse_spark_${TIMESTAMP}"
RUN_NAME="tooluse_spark_bs${DEVICE_BS}_acc${ACCUM_STEPS}_ng${NUM_GPUS}_eff${EFFECTIVE_BS}_numgen${NUM_GENERATIONS}"

if [ "${NUM_GPUS}" -gt 1 ]; then DIST_FLAGS="--multi_gpu"; else DIST_FLAGS=""; fi

HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 \
uv run accelerate launch --num_processes "${NUM_GPUS}" ${DIST_FLAGS} -m scripts.train \
  --output_dir "${RUN_DIR}" \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${DEVICE_BS}" \
  --gradient_accumulation_steps "${ACCUM_STEPS}" \
  --ref_model_mixup_alpha 0.02 \
  --eval_steps 20 \
  --eval_strategy steps \
  --per_device_eval_batch_size 8 \
  --save_steps 100 \
  --max_prompt_length 1024 \
  --max_completion_length 2048 \
  --num_generations "${NUM_GENERATIONS}" \
  --warmup_steps 10 \
  --run_name "${RUN_NAME}" \
  --paper_hparams
```

# Low-data SDFT (SuperGLUE small-data)

```bash
TASK=copa                  # one of: copa, cb, wsc
NUM_GPUS=1
CUDA_DEVICES=0
DEVICE_BS=2
ACCUM_STEPS=8
NUM_GENERATIONS=4
EFFECTIVE_BS=$((DEVICE_BS * ACCUM_STEPS * NUM_GPUS))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/home/$USER/sdft-experiments/runs/${TASK}_sdft_${TIMESTAMP}"
RUN_NAME="${TASK}_sdft_eff${EFFECTIVE_BS}_numgen${NUM_GENERATIONS}"

if [ "${NUM_GPUS}" -gt 1 ]; then DIST_FLAGS="--multi_gpu"; else DIST_FLAGS=""; fi

HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 \
uv run accelerate launch --num_processes "${NUM_GPUS}" ${DIST_FLAGS} -m scripts.train \
  --task "${TASK}" \
  --output_dir "${RUN_DIR}" \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${DEVICE_BS}" \
  --gradient_accumulation_steps "${ACCUM_STEPS}" \
  --ref_model_mixup_alpha 0.02 \
  --eval_steps 20 \
  --eval_strategy steps \
  --per_device_eval_batch_size 8 \
  --save_steps 100 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --num_generations "${NUM_GENERATIONS}" \
  --warmup_steps 10 \
  --run_name "${RUN_NAME}" \
  --paper_hparams
```

For these runs, checkpoint selection and validation should track `eval_small_data_accuracy`.
