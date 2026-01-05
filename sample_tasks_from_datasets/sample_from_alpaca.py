import json
import json
import argparse
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import random

parser=argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=50000, help="Number of prompts to process.")
parser.add_argument("--model", type=str, default="/root/hjh/LLM/qwen2.5-14B/", help="Name of the model.")
args = parser.parse_args()   
# 定义你的JSON文件名
file_path = '/root/hjh/datasets/alpaca_data.json'
tokenizer = get_tokenizer(args.model)
filtered_dataset: List[Tuple[int, str, int, int, str]] = []
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompt_id = 0
    for i, item in enumerate(data):
        instruction = item.get('instruction', '') 
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        # Tokenize the prompts and completions.
        if input_text == '':
            prompt = instruction
        else:
            prompt = f"{instruction}\n\n{input_text}"
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = output_text
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len + output_len > 131072: 
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt_id, prompt, prompt_len, output_len, 'alpaca'))
        prompt_id += 1
    filtered_dataset_sample = random.sample(filtered_dataset, prompt_id)

    data_filename = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/sampled_alpaca-qwen2-5-14b.json'

    with open(data_filename, 'w') as json_file:
        json.dump(filtered_dataset_sample, json_file, indent=None)
   
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请检查文件名和路径是否正确。")
except json.JSONDecodeError:
    print(f"错误：文件 '{file_path}' 的格式不是有效的 JSON。请检查文件内容。")
except Exception as e:
    print(f"发生了一个意料之外的错误: {e}")