# EfficientServe

A Large Model Inference Service System for Hybrid Application Scenarios

一个面向混合应用场景的高效大模型推理服务系统的设计与实现。

<img width="517" height="247" alt="image" src="https://github.com/user-attachments/assets/e31cc77f-d0bb-4fd4-a644-a350ba7f9820" />

## 如何开始

1. 安装好vLLM。安装参见：https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

2. 克隆并进入本项目：git clone https://github.com/chiahao-huang/EfficientServe.git  && cd EfficientServe

3. 执行命令替换vLLM中的部分组件：

   ```bash
   cd our_modifications 
   ./replace_modifications.sh
   ```

4. 安装自适应动态缓存分配机制的CUDA内核：

   ```bash
   cd adaptive_dynamic_cache_kernels
   python adaptive_dynamic_cache_kernels/adaptive_dynamic_cache_setup.py build_ext --inplace
   ```

至此，vLLM就变成了EfficientServe系统。

## 实验数据

可采用本人全部实验数据，链接为：https://github.com/chiahao-huang/EfficientServe/tree/main/sample_tasks_from_datasets/sampled_experiment_datasets

## 实验启动

关于如何配置启动推理系统和负载脚本。见这两个脚本文件：https://github.com/chiahao-huang/EfficientServe/blob/main/send_tasks.sh

https://github.com/chiahao-huang/EfficientServe/blob/main/generate_client_tasks.py

启动推理服务系统

```bash
python -m vllm.entrypoints.openai.api_server --model /root/hjh/LLM/qwen2.5-14B/ --enforce-eager --disable-log-requests --dtype 'float16' --gpu-memory-utilization 0.75
```

启动任务负载发生器

```bash
python plugin/EfficientServe/generate_client_tasks.py --model /root/hjh/LLM/qwen2.5-14B/ --request-rate 2 --cv 1 --request-type mixed
```

## 实验结果

> 在各种单一负载上EfficientServe在SLO达成率上表现都比其他系统好
>
> 在混合负载上，Efficient Serve表现依然由于其他现有推理服务系统
>
> 无论在哪种数据集上，在90%的达成率的目标下， EfficientServe系统所能承载的请求到达速率平均比vLLM高出2～4x，比Sarathi-Serve/FastGen高出1.8～3.5x；在60%的SLO达成率的目标下，EfficientServe系统所能承载的请求到达速率平局比比vLLM高出3.5～6x，比Sarathi-Serve/FastGen高出2.5~4x

<img width="1182" height="516" alt="image" src="https://github.com/user-attachments/assets/7a75b1d4-794f-4da9-a994-2b2bcc03f718" />
