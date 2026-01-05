python -m vllm.entrypoints.openai.api_server --model /root/hjh/LLM/opt-13b --enforce-eager --disable-log-requests --gpu-memory-utilization 0.75
python -m vllm.entrypoints.openai.api_server --model /root/hjh/LLM/qwen2.5-14B/ --enforce-eager --disable-log-requests --dtype 'float16' --gpu-memory-utilization 0.75
