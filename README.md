<div align="center">

<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
  <img src="./figures/logo.png" alt="EchoRL Icon" width="50">
  EchoRL: Reinforcement Learning via Rollout Echoing
</h1>


[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/yafuly)

</div>

---

# 📚 Overview
- 🎉 [News](#news)  
- 📖 [Introduction](#introduction)  
- ✨ [Getting Started](#getting-started)  
- 🔧 [Usage](#usage)  
- 📃 [Evaluation](#evaluation)  
- 🎈 [Citation](#citation)  
- 🌻 [Acknowledgement](#acknowledgement)  

---

# 🎉 News
- This repository hosts the official code for **EchoRL: Reinforcement Learning via Rollout Echoing** (ICML 2026 submission).

---

# 📖 Introduction

**EchoRL** addresses the **advantage degeneration** problem in Reinforcement Learning with Verifiable Rewards (RLVR) post-training. As training progresses, an increasing fraction of prompts become **advantage-degenerated**: all self-generated rollouts achieve verified-success (e.g., all receive reward `r=1`), causing the group standard deviation of rewards to collapse to zero. This makes each rollout's advantage degenerate to zero, leading to vanishing policy-gradient updates and marginal training gains.

## Key Insight

Even when rollouts share identical verifiable rewards, they may contain **qualitatively different reasoning paths** with distinct **step-level entropy patterns**. For example, some rollouts may use brute-force approaches while others employ elegant algorithmic insights (e.g., the log-derivative trick). These differences represent valuable learning signals that should be reinforced.

## EchoRL Solution

EchoRL is a plug-and-play module that:

1. **Identifies EchoClip**: From verified-success rollouts, EchoRL selects a prefix (EchoClip) ending at the step with the highest entropy, based on step-level entropy analysis.
2. **Provides Auxiliary Supervision**: EchoRL applies an auxiliary loss on these EchoClip prefixes, ensuring stable gradients even when group-relative advantages degenerate to zero.

This approach enables continued learning from advantage-degenerated prompts, improving RLVR post-training effectiveness across nine benchmarks, four LLM backbones, and seven popular RLVR methods.

## What This Repository Provides

- **GRPO Baseline**: Standard Group Relative Policy Optimization training script
- **EchoRL Implementation**: GRPO + entropy-based EchoClip supervision
- **Data Processing**: Scripts to prepare training data from OpenR1-Math
- **Evaluation Tools**: Scripts for math reasoning benchmarks (AIME, AMC, MATH-500, etc.)
- **Analysis Tools**: Golden trajectory entropy computation for EchoClip selection

---

# ✨ Getting Started

## Installation

### Basic Installation

```bash
# Create and activate conda environment
conda create -n echrl python=3.10
conda activate echrl

# Navigate to EchoRL repository root
cd /path/to/EchoRL

# Install EchoRL core dependencies
cd echrl
pip install -r requirements.txt
pip install -e .

# Install VERL (RL framework)
cd verl
pip install -r requirements.txt
pip install -e .
```

### Alternative Installation (if encountering dependency issues)

If you encounter issues with deprecated packages (e.g., `pyairports`), use this alternative installation:

```bash
conda create -n echrl python=3.10
conda activate echrl

# Install airports-py and outlines
pip install airports-py
git clone https://github.com/dottxt-ai/outlines.git
cd outlines
git checkout 0.0.46
pip install .
cd ..

# Install EchoRL
cd /path/to/EchoRL/echrl
pip install -r requirements.v2.txt  # if available
pip install -e .
cd verl
pip install -e .
```

### FlashAttention Installation

If you encounter issues installing `flash-attn`, install a pre-built wheel compatible with your CUDA/PyTorch version:

```bash
# Example for CUDA 12 + PyTorch 2.4
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Repository Structure

```
EchoRL/
├── echrl/                          # Core EchoRL code
│   ├── requirements.txt            # Python dependencies
│   ├── deepscaler/                 # Reward computation & utilities
│   │   ├── rewards/
│   │   │   ├── math_reward.py      # Math reward computation (RLVR)
│   │   │   └── math_utils/         # Math utilities
│   │   └── system_prompts.py       # System prompts for reasoning
│   └── verl/                       # VERL RL framework
│       └── verl/
│           └── mix_src/
│               ├── main_mix_ppo.py  # Main training entry point
│               └── config/          # Configuration files
├── data/                           # Data processing
│   ├── prepare_train.py            # Prepare training data (parquet)
│   └── test.parquet                # Example evaluation data
├── exp_scripts/                    # Training scripts
│   ├── baseline_train_pure_grpo.sh                    # GRPO baseline
│   └── baseline_train_pure_grpo_entropy_sft.sh        # EchoRL (GRPO + EchoClip)
└── eval_scripts/                   # Evaluation & analysis
    ├── generate_vllm.py            # vLLM-based generation
    ├── compute_golden_entropy.py   # Golden trajectory entropy analysis
    ├── oat_math_grader.py          # Math grading utilities
    └── collect_results.py          # Result aggregation
```

---

# 🔧 Usage

## Data Preparation

Prepare training data in parquet format:

```bash
cd data
python prepare_train.py
```

This script:
- Downloads `Elliott/Openr1-Math-46k-8192` from Hugging Face
- Converts to parquet format: `../data/openr1.parquet`

For validation, ensure `data/valid.parquet` exists (you can create a small held-out split from the training data).

## Training

### 1. GRPO Baseline

Train using standard GRPO (without EchoRL):

```bash
cd exp_scripts

# Set environment variables
export WANDB_API_KEY="your_api_key"
export WANDB_ENTITY="EchoRL"          # or your entity
export WANDB_PROJECT="GRPO"
export MODEL_PATH="/path/to/Qwen2.5-Math-7B"
export DATA_DIR="/path/to/EchoRL/data"
export EXP_NAME="GRPO_8192"

# Submit training job (SLURM)
sbatch baseline_train_pure_grpo.sh
```

**Key Configuration** (from `baseline_train_pure_grpo.sh`):
- `algorithm.adv_estimator=grpo`: Use GRPO advantage estimation
- `data.train_files=$DATA_DIR/openr1.parquet`: Training data
- `data.max_response_length=8192`: Maximum response length
- `actor_rollout_ref.rollout.n=8`: Sample 8 rollouts per prompt
- `actor_rollout_ref.actor.optim.lr=1e-6`: Learning rate

### 2. EchoRL (GRPO + EchoClip Supervision)

Train with EchoRL's entropy-based EchoClip supervision:

```bash
cd exp_scripts

# Set environment variables
export WANDB_API_KEY="your_api_key"
export WANDB_ENTITY="EchoRL"
export WANDB_PROJECT="EchoRL"
export MODEL_PATH="/path/to/Qwen2.5-Math-7B"
export DATA_DIR="/path/to/EchoRL/data"
export EXP_NAME="ECHORL_8192"

# Submit training job
sbatch baseline_train_pure_grpo_entropy_sft.sh
```

**EchoRL-Specific Configuration** (additional flags in `baseline_train_pure_grpo_entropy_sft.sh`):
- `+actor_rollout_ref.actor.use_entropy_sft_actor=True`: Enable entropy-based actor
- `+actor_rollout_ref.actor.use_entropy_sft=True`: Enable EchoClip supervision
- `+actor_rollout_ref.actor.entropy_sft_coef=0.001`: EchoClip loss coefficient
- `+actor_rollout_ref.actor.success_reward_value=1.0`: Success reward threshold
- `+actor_rollout_ref.actor.newline_token_ids=[198,271]`: Token IDs for step boundaries

These flags implement the **EchoClip mechanism**: identifying high-entropy reasoning steps in verified-success rollouts and applying auxiliary supervision on the corresponding prefixes.

### Training Output

Checkpoints are saved to:
```
results/checkpoints/$EXP_NAME/
```

Training logs are available via:
- Console output (redirected to `./results/train_results/`)
- Weights & Biases (if configured)

## Inference

Use a trained model for inference:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "your/echorl-or-grpo-model"

question = "Which number is larger? 9.11 or 9.9?"

tokenizer = AutoTokenizer.from_pretrained(model_path)
messages = [{"role": "user", "content": question}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_path)
params = SamplingParams(temperature=0.6, max_tokens=8192)
outputs = llm.generate([chat], params)
print(outputs[0].outputs[0].text)
```

---

# 📃 Evaluation

## Golden Trajectory Entropy Analysis

To reproduce the entropy analysis from the paper (used to identify EchoClip selection criteria):

```bash
python eval_scripts/compute_golden_entropy.py \
  --model_path /path/to/base-model \
  --data_path data/openr1.parquet \
  --output_dir /path/to/output/entropy_stats \
  --batch_size 1 \
  --max_length 4096 \
  --step 0 \
  --save_freq 100 \
  --dtype bfloat16 \
  --resume
```

This computes step-level entropy statistics on golden trajectories, which inform EchoClip selection (see paper Section 2 and Appendix H-I).

## Benchmark Evaluation

Evaluate on math reasoning benchmarks (AIME, AMC, MATH-500, ARC-c, GPQA, MMLU-Pro):

```bash
CUDA_VISIBLE_DEVICES=0 python eval_scripts/generate_vllm.py \
  --model_path /path/to/your/model \
  --input_file data/test.parquet \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file results/my_model_eval.jsonl \
  --template own
```

Then compute accuracy using grading utilities:

```bash
python eval_scripts/oat_math_grader.py \
  --input_file results/my_model_eval.jsonl \
  --output_file results/my_model_scores.json
```

**Note**: Some evaluation scripts (`eval_all.sh`, `eval_batch.sh`, etc.) contain project-specific paths. Update `ROOT`, `MODEL_PATH`, `DATA_DIR`, and conda environment names before use.

---

# 🎈 Citation

If you find EchoRL useful, please cite:

```bibtex
@inproceedings{echorl2026,
  title     = {EchoRL: Reinforcement Learning via Rollout Echoing},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

*(Please update with the final citation once the paper is published.)*

---

# 🌻 Acknowledgement

This repository builds on:
- **VERL**: The underlying RL framework for policy optimization
- **GRPO**: Group Relative Policy Optimization baseline
- **OpenR1-Math**: Training dataset for math reasoning

We thank the authors of these works for their valuable contributions.

---

## Key Concepts from the Paper

- **Advantage Degeneration**: When all rollouts for a prompt achieve verified-success, group reward variance collapses, causing advantages to degenerate to zero.
- **EchoClip**: A prefix of a verified-success rollout ending at the step with highest entropy, identified via step-level entropy analysis.
- **Entropy-Based Supervision**: Auxiliary loss applied to EchoClip prefixes to maintain learning signals when advantages degenerate.
- **Verified-Success Rollouts**: Rollouts that achieve verifiable reward (e.g., correct math solution), even if reasoning paths differ qualitatively.

For detailed theoretical analysis and experimental results, please refer to the paper.
