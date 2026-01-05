"""
基于vLLM的“benchmark_serving.py”脚本修改而来。

在服务器端，运行以下命令之一：
    vLLM OpenAI API服务器
    python -m vllm.entrypoints.openai.api_server \
        --model <你的模型> \
        --disable-log-requests


在客户端，运行：
    python generate_client_tasks.py \
        --backend <后端> \
        --model <你的模型> \
        --dataset-name sharegpt \
        --dataset-path <数据集路径> \
        --request-rate <请求速率> \ # 默认情况下<请求速率>为无穷大
        --num-prompts <提示词数量> #
"""
TASK_NUMBER = 200
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func_SLO import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float



async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    cv: int,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    
    num_reqs = len(input_requests)
    input_requests = iter(input_requests)
    gamma_shape = (1/cv)**2
    scale = 1/(request_rate*gamma_shape)
    
    
    intervals = []
    print('创建到达……')
    for i in range(num_reqs):
        interval = np.random.gamma(gamma_shape, scale)
        intervals.append(interval)
    intervals = np.array(intervals)
    count = 0
    
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
    
        # Sample the request interval from the exponential distribution.
        interval = intervals[count]
        count += 1
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

def calculate_SLO(all_result, ttft_slo, tbt_slo, num_requests):
     
    all_output_lens = np.array(all_result['output_lens'])
    all_ttfts = np.array(all_result['ttfts'])
    all_itls = all_result['itls']

    all_zero_index = np.argwhere(all_output_lens==0).squeeze()
    all_non_zero_index = ~ (all_output_lens==0)
    num_complete_requests = all_result['completed']
    num_valid_requests = sum(all_non_zero_index)
    #sum requests only get one output token, or their services face a timeout.
    gap_ = num_complete_requests-num_valid_requests 

    all_output_lens = all_output_lens[all_non_zero_index]
    all_ttfts = all_ttfts[all_non_zero_index]

    all_itls_new = []
    for i in range(len(all_non_zero_index)):
        if all_non_zero_index[i]:
            all_itls_new.append(all_itls[i])
    all_itls = all_itls_new

    all_itls_ = []
    all_itls_new = []
    all_ttfts_new = []
    for i in range(len(all_itls)):
        if len(all_itls[i]) > 1:
            all_itls_.append(np.percentile(np.sort(all_itls[i][1:]),99))
            all_itls_new.append(all_itls[i][1:])
        else:
            all_itls_.append(0)
            all_itls_new.append(0)
        all_ttfts_new.append(all_ttfts[i])


    all_itls_ = np.array(all_itls_)
    all_itls = all_itls_new
    all_ttfts = np.array(all_ttfts_new)

    SLO_attainment = sum((all_ttfts<ttft_slo) & (all_itls_<tbt_slo)) + gap_

    TTFT_attainment = sum(all_ttfts<ttft_slo) + gap_
    TBT_attainment = sum(all_itls_<tbt_slo) + gap_

    print('\n')
    print("{s:{c}^{n}}".format(s=' SLO 达成结果 ', n=50, c='='))
    print("{:<40} {:.2f}".format("Absolute SLO 达成数:",SLO_attainment))
    print("{:<40} {:.2f}".format("Absolute TTFT 达成数:",TTFT_attainment))
    print("{:<40} {:.2f}".format("Absolute P99 TBT 达成数:",TBT_attainment))
    print("{:<40} {:.2f}".format("SLO 达成率 (%):",SLO_attainment/num_requests*100))
    print("{:<40} {:.2f}".format("TTFT 达成率 (%):",TTFT_attainment/num_requests*100))
    print("{:<40} {:.2f}".format("TBT 达成率:",TBT_attainment/num_requests*100))
    print("=" * 50)
        
    

def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # 我们使用分词器来统计所有服务后端的输出令牌数量，而不是查看len(outputs[i].itl)，因为多个输出令牌可能被捆绑在一起。
            # 注意：这可能会略微增加输出的令牌数量
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][2]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "所有请求都失败了。这很可能是由于基准参数配置错误导致的。",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # 如果后端不支持流式传输，ttfts为空
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    cv: float,
    ttft_slo: float,
    tbt_slo: float,
    disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("开始初始的单提示词测试运行...")
    id, test_prompt, test_prompt_len, test_output_len, type = input_requests[0] # @黄佳浩
    num_requests = len(input_requests)
    test_input = RequestFuncInput(
        id = id,# @黄佳浩
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "初始测试运行失败 - 请确保基准测试参数已正确指定。 "
            f"错误: {test_output.error}")
    else:
        print("初始测试运行已完成。开始主要基准测试运行……")
    print(f"流量请求率: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, cv):
        id, prompt, prompt_len, output_len, type = request
        # output_len = 4096  # @黄佳浩
        request_func_input = RequestFuncInput(
            id = id,# @黄佳浩
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    
    calculate_SLO(result, ttft_slo, tbt_slo, num_requests)
    
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}:{args.port}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    
    
    input_requests = []
    max_out_len = 0
    base_path = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/'
    arxiv_path = base_path + 'sampled_arxiv-qwen2-5-14b.json'
    codegen_path = base_path + 'sampled_codegen-qwen2-5-14b.json'
    alpaca_path = base_path + 'sampled_alpaca-qwen2-5-14b.json'
    sharegpt_path = base_path + 'sampled_sharegpt-qwen2-5-14b.json'
    gsm8k_path = base_path + 'sampled_gsm8k-qwen2-5-14b.json'
    request_type = args.request_type
    # @黄佳浩：构建混合数据集
    if request_type == 'mixed':
        with open(arxiv_path) as f:
            arxiv_input_requests = json.load(f)
        with open(codegen_path) as f:
            codegen_input_requests = json.load(f)
        with open(alpaca_path) as f:
            alpaca_input_requests = json.load(f)
        with open(sharegpt_path) as f:
            sharegpt_input_requests = json.load(f)
        with open(gsm8k_path) as f:
            gsm8k_input_requests = json.load(f)
        for i in range(TASK_NUMBER):
            random_num = random.randint(1, 100)
            if random_num > 80:
                selected_idx = np.random.randint(0, TASK_NUMBER)
                request = arxiv_input_requests[selected_idx]
                request[0] = i
                max_out_len = max(max_out_len, request[3])
                input_requests.append(request)
            elif random_num >60:
                selected_idx = np.random.randint(0, TASK_NUMBER)
                request = alpaca_input_requests[selected_idx]
                request[0] = i
                max_out_len = max(max_out_len, request[3])
                input_requests.append(request)
            elif random_num > 40:
                selected_idx = np.random.randint(0, TASK_NUMBER)
                request = codegen_input_requests[selected_idx]
                request[0] = i
                max_out_len = max(max_out_len, request[3])
                input_requests.append(request)
            elif random_num > 20:
                selected_idx = np.random.randint(0, TASK_NUMBER)
                request = sharegpt_input_requests[selected_idx]
                request[0] = i
                max_out_len = max(max_out_len, request[3])
                input_requests.append(request)
            else :
                selected_idx = np.random.randint(0, TASK_NUMBER)
                request = gsm8k_input_requests[selected_idx]
                request[0] = i
                max_out_len = max(max_out_len, request[3])
                input_requests.append(request)
        # 将采样的请求再度写入json
        data_filename = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/sampled_mixed-qwen2-5-14b.json'
        with open(data_filename, 'w') as json_file:
            json.dump(input_requests, json_file, indent=None)   
    else:
        with open(alpaca_path) as f:
            input_requests = json.load(f)
        # 重新赋值index
        for i in range(TASK_NUMBER):
            input_requests[i][0] = i
            max_out_len = max(max_out_len, input_requests[i][3])

    # print(input_requests[0])
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            cv=args.cv,
            ttft_slo=args.ttft_slo,
            tbt_slo=args.tbt_slo,
            disable_tqdm=args.disable_tqdm,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "元数据格式无效。请使用KEY=VALUE格式。."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"result-{args.request_rate}qps-{base_model_id}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="对在线服务吞吐量进行基准测试.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="如果不使用HTTP主机和端口，则为服务器或API的基础URL.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API端点",
    )
    parser.add_argument(
        "--request-type",
        type=str,
        default="single",
        choices=["mixed","single"],
        help="用于进行服务模拟的数据集类型。",
    )
    
    parser.add_argument("--dataset-path",
                        type=str,
                        default='/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets',# @黄佳浩
                        help="数据集路径")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/hjh/LLM/qwen2.5-14B/",
        required=True,
        help="模型路径或名称",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "分词器的名称或路径（如果不使用默认分词器）。",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=TASK_NUMBER,
        help="要处理的提示词数量。",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "每个请求的输入标记数量，仅用于随机抽样。",
    )
    parser.add_argument(
        "--ttft-slo",
        type=int,
        default=1.0,
        help=
        "TTFT的SLO要求",
    )
    parser.add_argument(
        "--tbt-slo",
        type=int,
        default=1.0,
        help=
        "TBT的SLO要求",
    )
    
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "每个请求的输出令牌数量，仅用于随机采样。",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=3.0,
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=int("1"),
        help="伽马分布的协方差（用于控制突发性）",
    )
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任来自huggingface的远程代码",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="指定禁用tqdm进度条。",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="指定将基准测试结果保存到一个json文件中",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="指定用于保存基准测试JSON结果的目录。"
        "如果未指定，结果将保存在当前目录中。",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="指定用于保存基准测试JSON结果的文件名。"
        "如果未指定，结果将保存到 "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " 格式.",
    )

    args = parser.parse_args()
    main(args)
