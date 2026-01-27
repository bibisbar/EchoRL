#!/usr/bin/env python3
"""
Compute entropy for golden trajectories from the openr1 dataset.
This script loads the model, runs inference on golden trajectories,
and saves the per-token entropy in the same format as training entropy collection.
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits: -sum(p * log(p))"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def load_dataset(data_path: str, tokenizer, max_prompt_length: int = 1024, max_target_length: int = 2048):
    """Load dataset and extract prompts + golden trajectories."""
    df = pd.read_parquet(data_path)
    
    samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
        # Get prompt
        if 'prompt' in row:
            prompt_data = row['prompt']
        else:
            continue
        
        # Get target/golden trajectory
        if 'target' in row and row['target'] is not None:
            target_data = row['target']
        else:
            continue
        
        # Process prompt
        if isinstance(prompt_data, list):
            # Chat format
            prompt_text = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = str(prompt_data)
        
        # Process target
        if isinstance(target_data, dict):
            target_text = target_data.get('content', str(target_data))
        elif isinstance(target_data, list) and len(target_data) > 0:
            target_text = target_data[0].get('content', str(target_data[0])) if isinstance(target_data[0], dict) else str(target_data[0])
        else:
            target_text = str(target_data)
        
        # Skip empty targets
        if not target_text or target_text.strip() == '':
            continue
        
        # Handle <think> prefix
        if prompt_text.endswith('<think>\n') and target_text.startswith('<think>\n'):
            target_text = target_text[len('<think>\n'):]
        
        samples.append({
            'idx': idx,
            'prompt_text': prompt_text,
            'target_text': target_text,
        })
    
    print(f"Loaded {len(samples)} samples with valid golden trajectories")
    return samples


def process_single_sample(model, tokenizer, sample, max_length, device):
    """Process a single sample and return entropy."""
    prompt_text = sample['prompt_text']
    target_text = sample['target_text']
    
    # Tokenize prompt
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')['input_ids']
    
    # Tokenize target (response)
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors='pt')['input_ids']
    
    # Concatenate prompt + target
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    
    # Truncate if too long
    if full_ids.shape[1] > max_length:
        # Keep prompt, truncate target
        max_target_len = max_length - prompt_ids.shape[1]
        if max_target_len <= 0:
            return None
        target_ids = target_ids[:, :max_target_len]
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    
    prompt_length = prompt_ids.shape[1]
    target_length = target_ids.shape[1]
    
    # Move to device
    full_ids = full_ids.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=full_ids, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # Compute entropy for target tokens only
    target_logits = logits[0, prompt_length-1:prompt_length+target_length-1, :]
    
    # Compute per-token entropy
    per_token_entropy = compute_entropy_from_logits(target_logits).float().cpu().numpy()
    
    return {
        'sample_id': sample['idx'],
        'prompt': prompt_text,
        'response': target_text,
        'prompt_ids': prompt_ids[0].tolist(),
        'response_ids': target_ids[0].tolist(),
        'reward': None,
        'per_token_entropy': per_token_entropy,
    }


def process_batch(model, tokenizer, batch_samples, max_length, device):
    """Process a batch of samples with padding."""
    results = []
    
    # Prepare batch data
    all_full_ids = []
    all_prompt_lengths = []
    all_target_lengths = []
    valid_samples = []
    
    for sample in batch_samples:
        prompt_text = sample['prompt_text']
        target_text = sample['target_text']
        
        # Tokenize
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        
        # Concatenate
        full_ids = torch.cat([prompt_ids, target_ids], dim=0)
        
        # Truncate if needed
        if len(full_ids) > max_length:
            max_target_len = max_length - len(prompt_ids)
            if max_target_len <= 0:
                continue
            target_ids = target_ids[:max_target_len]
            full_ids = torch.cat([prompt_ids, target_ids], dim=0)
        
        all_full_ids.append(full_ids)
        all_prompt_lengths.append(len(prompt_ids))
        all_target_lengths.append(len(target_ids))
        valid_samples.append({
            'sample': sample,
            'prompt_ids': prompt_ids.tolist(),
            'target_ids': target_ids.tolist(),
        })
    
    if not valid_samples:
        return results
    
    # Pad to same length
    max_len = max(len(ids) for ids in all_full_ids)
    padded_ids = torch.zeros(len(all_full_ids), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(all_full_ids), max_len, dtype=torch.long)
    
    for i, ids in enumerate(all_full_ids):
        padded_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1
    
    # Move to device
    padded_ids = padded_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=padded_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # Extract entropy for each sample
    for i, (valid_sample, prompt_len, target_len) in enumerate(zip(valid_samples, all_prompt_lengths, all_target_lengths)):
        sample = valid_sample['sample']
        
        # Get logits for target tokens
        target_logits = logits[i, prompt_len-1:prompt_len+target_len-1, :]
        
        # Compute per-token entropy
        per_token_entropy = compute_entropy_from_logits(target_logits).float().cpu().numpy()
        
        results.append({
            'sample_id': sample['idx'],
            'prompt': sample['prompt_text'],
            'response': sample['target_text'],
            'prompt_ids': valid_sample['prompt_ids'],
            'response_ids': valid_sample['target_ids'],
            'reward': None,
            'per_token_entropy': per_token_entropy,
        })
    
    return results


def save_entropy_data(results, output_dir: str, step: int = 0):
    """Save entropy data in the same format as training entropy collection."""
    step_dir = os.path.join(output_dir, f'step_{step}')
    os.makedirs(step_dir, exist_ok=True)
    
    tensor_dir = os.path.join(step_dir, 'golden_trajectories_tensors')
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Prepare JSON data
    json_samples = []
    for idx, result in enumerate(results):
        # Save tensor
        tensor_file = f'sample_{result["sample_id"]}_entropy.npy'
        tensor_path = os.path.join(tensor_dir, tensor_file)
        np.save(tensor_path, result['per_token_entropy'])
        
        # Add to JSON data
        sample_info = {
            'sample_id': result['sample_id'],
            'step': step,
            'prompt': result['prompt'],
            'response': result['response'],
            'prompt_ids': result['prompt_ids'],
            'response_ids': result['response_ids'],
            'reward': result['reward'],
            'entropy_tensor_file': f'golden_trajectories_tensors/{tensor_file}',
        }
        json_samples.append(sample_info)
    
    # Save JSON
    json_data = {
        'step': step,
        'correct_rollouts': [],
        'wrong_rollouts': [],
        'golden_trajectories': json_samples,
        'stats': {
            'correct_rollouts': {'count': 0},
            'wrong_rollouts': {'count': 0},
            'golden_trajectories': {'count': len(json_samples)},
        }
    }
    
    json_file = os.path.join(step_dir, 'samples.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(json_samples)} samples to {step_dir}")


def append_results_to_file(results, output_dir: str, step: int = 0):
    """Append results incrementally - save tensors immediately, update JSON."""
    step_dir = os.path.join(output_dir, f'step_{step}')
    os.makedirs(step_dir, exist_ok=True)
    
    tensor_dir = os.path.join(step_dir, 'golden_trajectories_tensors')
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Save each tensor immediately
    saved_info = []
    for result in results:
        tensor_file = f'sample_{result["sample_id"]}_entropy.npy'
        tensor_path = os.path.join(tensor_dir, tensor_file)
        np.save(tensor_path, result['per_token_entropy'])
        
        saved_info.append({
            'sample_id': result['sample_id'],
            'step': step,
            'prompt': result['prompt'],
            'response': result['response'],
            'prompt_ids': result['prompt_ids'],
            'response_ids': result['response_ids'],
            'reward': result['reward'],
            'entropy_tensor_file': f'golden_trajectories_tensors/{tensor_file}',
        })
    
    # Append to JSONL file (one JSON per line for incremental saving)
    jsonl_file = os.path.join(step_dir, 'samples.jsonl')
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        for info in saved_info:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')
    
    return len(saved_info)


def finalize_json(output_dir: str, step: int = 0):
    """Convert JSONL to final JSON format."""
    step_dir = os.path.join(output_dir, f'step_{step}')
    jsonl_file = os.path.join(step_dir, 'samples.jsonl')
    json_file = os.path.join(step_dir, 'samples.json')
    
    if not os.path.exists(jsonl_file):
        print(f"No JSONL file found at {jsonl_file}")
        return
    
    # Read all samples from JSONL
    json_samples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                json_samples.append(json.loads(line))
    
    # Create final JSON structure
    json_data = {
        'step': step,
        'correct_rollouts': [],
        'wrong_rollouts': [],
        'golden_trajectories': json_samples,
        'stats': {
            'correct_rollouts': {'count': 0},
            'wrong_rollouts': {'count': 0},
            'golden_trajectories': {'count': len(json_samples)},
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinalized {len(json_samples)} samples to {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute entropy for golden trajectories')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset (parquet)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for entropy data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--step', type=int, default=0, help='Step number for output directory naming')
    parser.add_argument('--save_freq', type=int, default=100, help='Save frequency (every N batches)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype')
    parser.add_argument('--resume', action='store_true', help='Resume from existing progress')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}")
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
    )
    model.eval()
    print(f"Model loaded with dtype {args.dtype}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    samples = load_dataset(args.data_path, tokenizer)
    
    # Limit samples if specified
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")
    
    # Check for resume
    start_idx = 0
    if args.resume:
        step_dir = os.path.join(args.output_dir, f'step_{args.step}')
        jsonl_file = os.path.join(step_dir, 'samples.jsonl')
        if os.path.exists(jsonl_file):
            with open(jsonl_file, 'r') as f:
                start_idx = sum(1 for _ in f)
            print(f"Resuming from sample {start_idx}")
    
    # Process in batches with incremental saving
    print(f"Computing entropy for golden trajectories (batch_size={args.batch_size}, save_freq={args.save_freq})...")
    
    total_saved = start_idx
    batch_results = []
    
    pbar = tqdm(range(start_idx, len(samples)), desc="Computing entropy", initial=start_idx, total=len(samples))
    
    for i in pbar:
        sample = samples[i]
        
        # Process single sample (safer than batch for variable length)
        result = process_single_sample(model, tokenizer, sample, args.max_length, device)
        
        if result is not None:
            batch_results.append(result)
        
        # Save periodically
        if len(batch_results) >= args.save_freq:
            saved = append_results_to_file(batch_results, args.output_dir, step=args.step)
            total_saved += saved
            batch_results = []
            pbar.set_postfix({'saved': total_saved})
    
    # Save remaining results
    if batch_results:
        saved = append_results_to_file(batch_results, args.output_dir, step=args.step)
        total_saved += saved
    
    # Finalize JSON
    finalize_json(args.output_dir, step=args.step)
    
    print(f"\nDone! Total samples saved: {total_saved}")


if __name__ == '__main__':
    main()
