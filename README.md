# EfficientServe

A Large Model Inference Service System for Hybrid Application Scenarios

## How to start

1. Install vLLM properly. For installation instructions, refer to: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

2. Clone and enter this project: git clone https://github.com/chiahao-huang/EfficientServe.git && cd EfficientServe

3. Execute the command to replace some components in vLLM:

   ```bash
   cd our_modifications 
   ./replace_modifications.sh
   ```

4. Install the CUDA kernel with adaptive dynamic cache allocation mechanism:

   ```bash
   cd adaptive_dynamic_cache_kernels
   python adaptive_dynamic_cache_kernels/adaptive_dynamic_cache_setup.py build_ext --inplace
   ```

At this point, vLLM has become the EfficientServe system.

## Experimental data

All of my experimental data can be used, and the link is: https://github.com/chiahao-huang/EfficientServe/tree/main/sample_tasks_from_datasets/sampled_experiment_datasets

##  Experiment Reproduction

Regarding how to configure and start the inference system and the load script. Please refer to these two script files.：https://github.com/chiahao-huang/EfficientServe/blob/main/send_tasks.sh

https://github.com/chiahao-huang/EfficientServe/blob/main/generate_client_tasks.py

**Start the inference service system**

```bash
python -m vllm.entrypoints.openai.api_server --model /root/hjh/LLM/qwen2.5-14B/ --enforce-eager --disable-log-requests --dtype 'float16' --gpu-memory-utilization 0.75
```

Start the task load generator

```bash
python plugin/EfficientServe/generate_client_tasks.py --model /root/hjh/LLM/qwen2.5-14B/ --request-rate 2 --cv 1 --request-type mixed
```

## Results

### Length Prediction Accuracy and Overhead

> EmPre achieves a significantly lower Mean Absolute Error (MAE) across all datasets compared to the BERT-based method. The prediction error of EmPre without the Bayesian smoothing mechanism is greater than that of EmPre, which indicates that the Bayesian smoothing mechanism has improved the prediction performance of the model. Meanwhile, we observe that the ratio of the predicted value to the groud truth decreases monotonically as the generation process progresses, as shown in Figure 8. This indicates that iterative prediction is very necessary because it can provide us with increasingly accurate predictions. Crucially, the EmPre MLP contains only 6.3M parameters. The resultsin Fig. 8 indicate that inference overhead is between 0.6ms and 1.2ms, which differs by several orders of magnitude from the time taken for a single forward propagation inference of a llm, hardly affects the forward propagation of the main model.
<img width="345" height="185" alt="image-20260302215450148" src="https://github.com/user-attachments/assets/7ec2685e-7ab3-4fce-b8c5-92229a91ebf0" style="zoom: 50%;"/>
<img width="347" height="199" alt="image-20260302215515135" src="https://github.com/user-attachments/assets/68f0c156-7db5-4dc0-8ab4-74356a33394b" style="zoom: 50%;"/>

<img width="354" height="185" alt="image-20260302215627783" src="https://github.com/user-attachments/assets/decc8d1e-de50-47b7-b5a4-ba87bd87ebfb" style="zoom: 50%;"/>
<img width="292" height="166" alt="image-20260302215716941" src="https://github.com/user-attachments/assets/ba8b27d8-f5c5-4a72-b850-e1391e9bb05f" style="zoom: 50%;"/>


### Overall system performance

> We evaluate the systems under the mixed workload using a Poisson arrival process. Traditional systems suffer from severe Head-of-Line blocking when long-sequence throughput tasks monopolize resources, starving interactive tasks. Fig. 9(a) demonstrates that EfficientServe effectively mitigates this issue. Under a strict 90% SLO attainment target, EfficientServe supports a maximum request arrival rate that is 2.6x that of vLLM, 2.1x of Sarathi-Serve, and 1.9x of FastGen. When the system operates under extreme overload conditions (e.g., targeting a 60% SLO), EfficientServe’s throughput advantage expands up to 4.8x over vLLM. This massive gain is attributed to the H Cache mechanism,which virtually doubles the available context capacity, allowing the QoS-aware scheduler to schedule tasks more aggressively. Furthermore, considering the particularity of mixed loads, we also conducted separate experimental analyses on single loads, the results are also shown in Fig. 9 (b) ˜ Fig.9(f).

![image-20260302220145596](/Users/roy/Library/Application Support/typora-user-images/image-20260302220145596.png)

> Real-world traffic rarely follows a smooth Poisson distribution. To test system resilience, we replay traces from BurstGPT, characterized by long-tail distributions and sudden concurrency spikes. For the convenience of the experiment, we sampled 108 seconds of data from the high-burst interval of the open-source BurstGPT dataset as the experimental load. Its burst fluctuations are shown in Fig 10. Meanwhile, we also analyzed the input and output distribution of the sampled load, and the results are shown in Fig 11. Under the BurstGPT trace, baseline systems experience catastrophic cascading violations during traffic surges due to hard OOM limits. In contrast, EfficientServe dynamically compresses low-priority KV Caches into H Caches, absorbing the

### Robustness Test

> Real-world traffic rarely follows a smooth Poisson distribution. To test system resilience, we replay traces from BurstGPT, characterized by long-tail distributions and sudden concurrency spikes. For the convenience of the experiment,  we sampled 108 seconds of data from the high-burst interval of the open-source BurstGPT dataset as the experimental load. Its burst fluctuations are shown in Fig 10. Meanwhile, we also analyzed the input and output distribution of the sampled load, and the results are shown in Fig 11. Under the BurstGPT trace, baseline systems experience catastrophic cascading violations during traffic surges due to hard OOM limits. In contrast, EfficientServe dynamically compresses low-priority KV Caches into H Caches, absorbing the traffic shock. As depicted in Fig. 12, EfficientServe surpasses the SLO attainment of vLLM, Sarathi-Serve, and FastGen by 22%, 17%, and 15%, respectively, proving its readiness for industrial deployment.

<img src="/Users/roy/Library/Application Support/typora-user-images/image-20260302220250510.png" alt="image-20260302220250510" style="zoom:50%;" /><img src="/Users/roy/Library/Application Support/typora-user-images/image-20260302220314586.png" alt="image-20260302220314586" style="zoom:50%;" />

### Ablation study

> Impact of Adaptive Chache Allocation: We first evaluate the system without the H Cache mechanism (denoted as EfficientServe-w/o-H). Under high-burst scenarios (e.g.,Coefficient of Variation = 5), the absence of memory elasticity leads to frequent Out-Of-Memory triggers. Consequently, the system is forced to systematically evict and recompute tasks, resulting in a significant drop in SLO attainment (by over 5.5%) and a sharp increase in tail latency.
>
> Impact of QoS-Aware Scheduling: Next, we replace our marginal-gain QoS scheduler with a standard FCFS strategy (denoted as EfficientServe-FCFS) while retaining the H Cache pool. As illustrated in Fig. 14, without the predictive interleaving of tasks, long throughput-oriented requests severely block interactive queries. This Head-of-Line (HoL) blocking causes TTFT violations to skyrocket, demonstrating that intelligent task scheduling is indispensable for mixed workloads.

<img src="/Users/roy/Library/Application Support/typora-user-images/image-20260302220727694.png" alt="image-20260302220727694" style="zoom:50%;" /><img src="/Users/roy/Library/Application Support/typora-user-images/image-20260302220737476.png" alt="image-20260302220737476" style="zoom:50%;" />
