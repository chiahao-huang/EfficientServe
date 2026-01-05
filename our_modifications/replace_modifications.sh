#!/bin/bash

# Get the vllm package source directory using Python
vllm_dir="/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/vllm"

# Check if the directory was successfully located
if [ -z "$vllm_dir" ]; then
  echo "vllm package not found. Exiting."
  exit 1
fi

echo "vllm package found at: $vllm_dir"


# Replace sequence and block part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/efficientserve_block.py" "$vllm_dir/block.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/efficientserve_sequence.py" "$vllm_dir/sequence.py"


#Replace attention part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/attention/efficientserve_layer.py" "$vllm_dir/attention/layer.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/attention/backends/efficientserve_abstract.py" "$vllm_dir/attention/backends/abstract.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/attention/backends/efficientserve_flash_attn.py" "$vllm_dir/attention/backends/flash_attn.py"

#Replace engine part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/engine/efficientserve_llm_engine.py" "$vllm_dir/engine/llm_engine.py"

#Replace core part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/core/efficientserve_block_manager.py" "$vllm_dir/core/block_manager_v1.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/core/efficientserve_interfaces.py" "$vllm_dir/core/interfaces.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/core/efficientserve_scheduler.py" "$vllm_dir/core/scheduler.py"

#Replace worker part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/worker/efficientserve_cache_engine.py" "$vllm_dir/worker/cache_engine.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/worker/efficientserve_model_runner.py" "$vllm_dir/worker/model_runner.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/worker/efficientserve_worker.py" "$vllm_dir/worker/worker.py"

#Replace model_executor part
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/model_executor/layers/efficientserve_linear.py" "$vllm_dir/model_executor/layers/linear.py"
cp "/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/our_modifications/model_executor/models/efficientserve_opt.py" "$vllm_dir/model_executor/models/opt.py"
