import json
import json
import argparse
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import random

parser=argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
parser.add_argument("--model", type=str, default="/root/hjh/LLM/Qwen2.5-14B-Instruct/", help="Name of the model.")
args = parser.parse_args()   
# 定义你的JSON文件名
file_path = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/gsm8k.json'
tokenizer = get_tokenizer(args.model)
filtered_dataset: List[Tuple[int, str, int, int, str]] = []
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    index = 0
    for i, item in enumerate(data):
        instruction = item.get('question', '') 
        input_text = item.get('input_', '')
        output_text = item.get('answer', '')
        # Tokenize the prompts and completions.
        if input_text == '':
            prompt = instruction
        else:
            prompt = f"{instruction}\n\nthe input is: {input_text}"
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = output_text
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 131072: # 切换成对应模型的最大上下文，opt是2048
            # Prune too long sequences.
            continue
        filtered_dataset.append((index, prompt, prompt_len, output_len, 'gsm8k'))
        index += 1
    filtered_dataset_sample = random.sample(filtered_dataset, index)

    data_filename = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/sampled_gsm8k-qwen2-5-14b.json'

    with open(data_filename, 'w') as json_file:
        json.dump(filtered_dataset_sample, json_file, indent=None)
   
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请检查文件名和路径是否正确。")
except json.JSONDecodeError:
    print(f"错误：文件 '{file_path}' 的格式不是有效的 JSON。请检查文件内容。")
except Exception as e:
    print(f"发生了一个意料之外的错误: {e}")