#!/bin/bash

source ~/.bashrc

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="your_wandb_entity"
export WANDB_PROJECT="your_wandb_project"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS


if ! conda activate echrl; then
echo "[ERROR] Conda environment 'echrl' not found."
conda info --envs || true
exit 1
fi

# NOTE: change to your root dir
ROOT=/path/to/EchoRL

export PYTHONPATH="$ROOT/echrl:${PYTHONPATH}"

ray stop --force || true

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Avoid forcing a specific attention backend on H100; let vLLM choose
unset VLLM_ATTENTION_BACKEND

export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Math-7B}"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export EXP_NAME="${EXP_NAME:-ECHORL_quicktest}"


cd $ROOT

echo "Using Python: $(which python)"
python -V
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import torch; print('torch =', torch.__version__)" || echo "[WARN] torch not found in selected Python"

DATA="${DATA:-$ROOT/data/valid.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/}"
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_NAME="${MODEL_NAME:-ECHORL_quicktest}"

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log