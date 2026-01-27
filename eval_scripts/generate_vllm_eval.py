#export HF_ENDPOINT=https://hf-mirror.com  
import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from math_verify import parse, verify
from oat_math_grader import boxed_reward_fn as oat_evaluate

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")
            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        return decorator

@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def make_conv_prime_sft(question, tokenizer):
    # for math problem
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end." 
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )

def generate_vllm_batch(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1, batch_size=32, dec_output_path=None, answers=None):
    """
    批量处理版本的 generate_vllm，用于 evaluation
    批量处理可以显著提升速度，同时保持容错机制
    支持边生成边写入，避免阻塞
    """
    # vllm模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
    print('number of GPUs: ', torch.cuda.device_count())
    
    # Ensure max context length is set explicitly to avoid backend block-table mismatches
    # 首先从模型 config 获取 max_position_embeddings（这是最准确的）
    max_model_len = None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        # 优先使用 max_position_embeddings（这是模型实际支持的最大长度）
        if hasattr(config, 'max_position_embeddings') and config.max_position_embeddings is not None:
            max_model_len = int(config.max_position_embeddings)
            print(f'Found max_position_embeddings from config: {max_model_len}')
        elif hasattr(config, 'model_max_length') and config.model_max_length is not None:
            max_model_len = int(config.model_max_length)
            print(f'Found model_max_length from config: {max_model_len}')
    except Exception as e:
        print(f'Warning: Failed to load config: {e}')
    
    # 如果 config 没有，尝试从 tokenizer 获取
    if max_model_len is None:
        max_model_len = getattr(tokenizer, "model_max_length", None)
        try:
            max_model_len = int(max_model_len) if max_model_len is not None else None
            if max_model_len:
                print(f'Found model_max_length from tokenizer: {max_model_len}')
        except Exception:
            max_model_len = None
    
    # 如果还是没有，使用默认值，但不要超过 16384（大多数模型的常见值）
    if max_model_len is None or max_model_len > 1000000:
        max_model_len = 16384  # 使用更保守的默认值
        print(f'Using default max_model_len: {max_model_len}')
    
    # 确保不超过模型的实际限制（16384 是大多数模型的常见值）
    # 如果检测到的值太大，可能是错误的，使用更保守的值
    if max_model_len > 16384:
        print(f'Warning: max_model_len {max_model_len} seems too large, capping at 16384')
        max_model_len = 16384
    
    print(f'Final max_model_len: {max_model_len}')
    
    # 构建 LLM 参数
    # 注意：vLLM 可能会自动检测 max_model_len，但我们需要显式设置以确保不超过模型限制
    llm_kwargs = {
        'model': model_path,
        'tensor_parallel_size': torch.cuda.device_count(),
        'gpu_memory_utilization': 0.85,
    }
    
    # 显式设置 max_model_len，确保不超过模型的实际限制
    if max_model_len is not None:
        llm_kwargs['max_model_len'] = max_model_len
        print(f'Setting max_model_len to {max_model_len}')

    llm = LLM(**llm_kwargs)

    # 第一步：准备所有 prompts
    print(f'Preparing {len(messages)} prompts...')
    gen_prompts = []
    for i, cur_message in enumerate(messages):
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
            )
            if i == 0:  
                print('Example input: ', gen_prompt)
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[0]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(cur_message[0]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: 
            raise ValueError(f'Invalid template: {template}')

        if i == 0:
            print('Example input: ', gen_prompt)
        
        gen_prompts.append(gen_prompt)

    # 第二步：批量处理，边生成边写入
    valid_outputs = []
    valid_to_original = []  # 记录有效输出对应的原始索引
    skipped_indices: list[int] = []
    total_batches = (len(gen_prompts) + batch_size - 1) // batch_size
    
    print(f'Processing {len(gen_prompts)} samples in {total_batches} batches (batch_size={batch_size})...')
    from tqdm import tqdm
    
    # 如果提供了输出路径，打开文件准备写入
    fo = None
    if dec_output_path:
        fo = open(dec_output_path, 'w')
        print(f'Writing results to {dec_output_path} in real-time...')
    
    try:
        for batch_idx in tqdm(range(0, len(gen_prompts), batch_size), desc="Batches"):
            batch_end = min(batch_idx + batch_size, len(gen_prompts))
            batch_prompts = gen_prompts[batch_idx:batch_end]
            batch_indices = list(range(batch_idx, batch_end))
            
            try:
                # 尝试批量生成
                batch_outputs = llm.generate(batch_prompts, sampling_params)
                # vLLM 返回一个列表，每个元素对应一个 prompt 的输出
                for local_idx, out in enumerate(batch_outputs):
                    original_idx = batch_indices[local_idx]
                    valid_outputs.append(out)
                    valid_to_original.append(original_idx)
                    
                    # 边生成边写入
                    if fo and answers:
                        prompt = out.prompt
                        for j in range(n):
                            generated_text = out.outputs[j].text
                            item = {
                                'prompt': prompt,
                                'generated_text': generated_text,
                                'answer': answers[original_idx],
                                'skipped': False
                            }
                            fo.write(json.dumps(item) + '\n')
                            fo.flush()  # 立即刷新到磁盘
            except Exception as e:
                # 批量失败，回退到逐样本处理这个 batch
                err_msg = str(e)
                print(f"[WARN] Batch {batch_idx}-{batch_end} failed, falling back to per-sample: {err_msg}")
                for i, prompt in zip(batch_indices, batch_prompts):
                    try:
                        out = llm.generate([prompt], sampling_params)
                        valid_outputs.append(out[0])
                        valid_to_original.append(i)
                        
                        # 边生成边写入
                        if fo and answers:
                            prompt_text = out[0].prompt
                            for j in range(n):
                                generated_text = out[0].outputs[j].text
                                item = {
                                    'prompt': prompt_text,
                                    'generated_text': generated_text,
                                    'answer': answers[i],
                                    'skipped': False
                                }
                                fo.write(json.dumps(item) + '\n')
                                fo.flush()
                    except Exception as e2:
                        err_msg2 = str(e2)
                        print(f"[WARN] skipping sample {i} due to generation error: {err_msg2}")
                        skipped_indices.append(i)
    finally:
        if fo:
            fo.close()
    
    print(f'Batch processing completed: {len(valid_outputs)} successful, {len(skipped_indices)} skipped')
    return valid_outputs, skipped_indices, valid_to_original

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, add_oat_evaluate=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False, batch_size=32):

    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'

    if force_generate or (not os.path.exists(dec_output_path)):
        # 数据处理
        messages = df['prompt'].tolist()
        
        assert remove_system is True
        if remove_system:
            print('remove system')
            assert messages[0][0]['role'] == 'system'
            messages = [message[1:] for message in messages]
            
        else:
            assert remove_system is False
            print('not remove system')
            
        answers = df['reward_model'].tolist()
        answers = [answer['ground_truth'] for answer in answers]
        # if debug:
            # answers = answers[:10]
        assert len(messages) == len(answers)
                
        print(messages[0])
        print(f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}, batch_size: {batch_size}")
        # 传递 dec_output_path 和 answers，实现边生成边写入
        outputs, skipped_indices, valid_to_original = generate_vllm_batch(
            messages, model_path, 
            template=template, 
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=max_tokens, 
            n=n, 
            batch_size=batch_size,
            dec_output_path=dec_output_path,  # 传递输出路径，边生成边写入
            answers=answers  # 传递答案，用于写入文件
        )
        total_cnt = len(messages)
        success_cnt = len(outputs)
        skip_cnt = len(skipped_indices)
        print(f"total: {total_cnt}, success: {success_cnt}, skipped: {skip_cnt}")
        # decoded.jsonl 已经在 generate_vllm_batch 中实时写入了，不需要重复写入
                    
        # format sort prompts, outputs, answers
        assert len(outputs[0].outputs) == n
        prompts = [out.prompt for out in outputs for j in range(n)]
        answers = [answers[valid_to_original[i]] for i in range(len(outputs)) for j in range(n)]  # 使用原始索引
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]
        # 保存原始索引映射，用于后续评分
        valid_to_original_expanded = [valid_to_original[i] for i in range(len(valid_to_original)) for j in range(n)]
    else:
        print('Found already decoded file, skip decoding...')
        jss = []
        with open(dec_output_path, 'r') as f:
            for line in f:
                jss.append(json.loads(line))
        
        # When resuming, compute basic stats as well
        total_cnt = len(df)
        success_cnt = len([item for item in jss if not item.get('skipped', False)])
        skip_cnt = total_cnt - success_cnt
        print(f"total: {total_cnt}, success: {success_cnt}, skipped: {skip_cnt}")

        outputs = [item['generated_text'] for item in jss if not item.get('skipped', False)]
        prompts = [item['prompt'] for item in jss if not item.get('skipped', False)]
        answers = [item['answer'] for item in jss if not item.get('skipped', False)]
        # 从 decoded 文件恢复时，无法知道原始索引，假设都是连续的
        valid_to_original_expanded = list(range(len(outputs)))
    
    data_sources = df['data_source'].tolist()
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0
    from tqdm import tqdm

    print('Scoring...')
    if skip_scoring:
        return
    
    # for i, output in tqdm(enumerate(outputs)):
    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        # print(i)
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        think_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            think_format = True
        if no_split_think:
            think_format = False

        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                labels = [False]
                
        if labels is None:
            try:
                labels = labeling_responses([generated_text,], answer)
            except Exception as e:
                labels = [False]
        
        if add_oat_evaluate:
            new_label = oat_evaluate(generated_text, answer, fast=False)
            new_label = new_label[1] == 1.0
            if any_true is True:
                if labels[0] is False and new_label is True:
                    diff_cnt += 1
                    # breakpoint()
                labels = [labels[0] or new_label]
            else:
                labels = [new_label]
        
        original_idx = valid_to_original_expanded[i] if i < len(valid_to_original_expanded) else i
        rets[data_sources[original_idx]].append(labels[0])
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg += 1

    
    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)
    
    accs = []
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
        accs.append(acc)
    
    print('avg acc: ', np.array(accs).mean())
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')

if __name__ == "__main__":
    import fire
    fire.Fire(main)

