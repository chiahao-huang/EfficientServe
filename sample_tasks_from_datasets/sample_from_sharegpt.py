import json
import argparse
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import random

parser=argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
parser.add_argument("--model", type=str, default="/root/hjh/LLM/qwen2.5-14B/", help="Name of the model.")

args = parser.parse_args()   

dataset_path = '/root/hjh/datasets/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json'
with open(dataset_path) as f:
    dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
dataset = [data for data in dataset if len(data["conversations"]) >= 2]
# Only keep the first two turns of each conversation.
dataset = [(data["conversations"][0]["value"],
            data["conversations"][1]["value"]) for data in dataset] 

random.shuffle(dataset)

filtered_dataset: List[Tuple[int, str, int, int, str]] = []
tokenizer = get_tokenizer(args.model)
index = 0
for i in range(len(dataset)):
    # Tokenize the prompts and completions.
    prompt = dataset[i][0]
    prompt_token_ids = tokenizer(prompt).input_ids
    completion = dataset[i][1]
    completion_token_ids = tokenizer(completion).input_ids
    prompt_len = len(prompt_token_ids)
    output_len = len(completion_token_ids)
    if prompt_len < 4 or output_len < 4:
        # Prune too short sequences.
        continue
    if prompt_len + output_len > 131072: # 切换成对应模型的最大上下文，opt是2048
        # Prune too long sequences.
        continue
    filtered_dataset.append((index, prompt, prompt_len, output_len, 'sharegpt'))
    index += 1
filtered_dataset_sample = random.sample(filtered_dataset, 500)

data_filename = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/sampled_sharegpt-qwen2-5-14b.json'

with open(data_filename, 'w') as json_file:
    json.dump(filtered_dataset_sample, json_file, indent=None)
   
