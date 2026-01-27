#!/bin/bash

source ~/.bashrc

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if ! conda activate echrl; then
    echo "[ERROR] Conda environment 'echrl' not found."
    conda info --envs || true
    exit 1
fi

# NOTE: change to your root dir
ROOT=/path/to/EchoRL

export PYTHONPATH="$ROOT/echrl:${PYTHONPATH}"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

cd $ROOT

echo "Using Python: $(which python)"
python -V
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import torch; print('torch =', torch.__version__)" || echo "[WARN] torch not found in selected Python"

# ============================================================================
# Configuration
# ============================================================================
export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Math-7B}"
DATA_PATH="${DATA_PATH:-$ROOT/data/openr1.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/entropy_data/golden_trajectory_entropy}"

# Optional: limit number of samples for testing
# MAX_SAMPLES=1000

echo "=========================================="
echo "Golden Trajectory Entropy Computation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run the script
python eval_scripts/compute_golden_entropy.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --max_length 4096 \
    --step 0 \
    --save_freq 100 \
    --dtype bfloat16 \
    --resume
    # --max_samples $MAX_SAMPLES  # Uncomment to limit samples

echo "Done!"

