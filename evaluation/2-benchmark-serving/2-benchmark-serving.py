"""Benchmark online serving throughput.
"""

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional
import os
import sys
import statistics
import aiohttp
import numpy as np
from tqdm import tqdm

from structs import TestRequest, Dataset, RequestResult
from backends import BACKEND_TO_PORTS

pbar: Optional[tqdm] = None

def sample_requests(dataset_path: str, num_prompts: int) -> List[TestRequest]:
    """
    sample_requests: 从数据集中采样指定数量的请求。
    如果请求数量超过数据集大小，则循环使用数据集中的请求。
    """
    dataset = Dataset.load(dataset_path)
    total_requests = len(dataset.reqs)
    
    if num_prompts > total_requests:
        print(f"Warning: Number of prompts ({num_prompts}) is larger than the dataset size ({total_requests}), cycling through the dataset.")

    sampled_requests = []
    for i in range(num_prompts):
        sampled_requests.append(dataset.reqs[i % total_requests])
    return sampled_requests

async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
    burst_count: int = 0  # 新增参数：前 burst_count 个请求立即发送
) -> AsyncGenerator[TestRequest, None]:
    total_requests = len(input_requests)
    input_requests_iter = iter(input_requests)

    # 对剩余请求部分计算延时
    remaining_count = total_requests - burst_count
    if request_rate not in [float("inf"), 0.0] and remaining_count > 0:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(remaining_count)]
        elif process_name in ["gamma", "possion"]:
            # 使用 gamma 分布来模拟泊松过程，这里保持 cv=1
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=remaining_count)
        else:
            raise ValueError(
                f"Unsupported process name: {process_name}, we currently support uniform, gamma and possion."
            )
    else:
        intervals = [0] * remaining_count

    for idx, request in enumerate(input_requests_iter):
        yield request
        # 前 burst_count 个请求立即发出，不等待延时
        if idx < burst_count:
            continue
        # 对剩余请求，等待对应延时（注意调整下标）
        interval_idx = idx - burst_count
        await asyncio.sleep(intervals[interval_idx])

async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    verbose: bool
) -> RequestResult:
    global pbar
    if backend == "deepspeed":
        payload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "min_new_tokens": output_len,
            "max_new_tokens": output_len,
            "stream": True,
            "max_length": int((prompt_len + output_len) * 1.2 + 10)  # *1.2 防止 tokenization 错误
        }
        
        request_start_time = time.perf_counter()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3*3600)) as session:
            token_timestamps = []
            generated_text = ""
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for data in response.content.iter_any():
                            token_timestamps.append(time.perf_counter())
                            try:
                                generated_text += json.loads(data.decode("utf-8")[6:])["text"][0]
                            except:
                                generated_text += data.decode("utf-8")
                    else:
                        print(response)
                        print(response.status)
                        print(response.reason)
                        sys.exit(1)
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
                print(e)
                sys.exit(1)
        request_end_time = time.perf_counter()
        
        if verbose:
            print(f"Prompt: {prompt}, Output: {generated_text}")
        
        pbar.update(1)
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=token_timestamps,
            lifetime_events=None
        )
    else:
        headers = {"User-Agent": "Benchmark Client"}
        if backend in ["distserve", "vllm"]:
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "temperature": 0.0 if use_beam_search else 1.0,
                "top_p": 1.0,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": False,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # 输入和输出的 token 数量和不能超过 2048（嵌入表大小的限制）
        assert prompt_len + output_len < 2048
        
        request_start_time = time.perf_counter()
        request_output = None

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                try:
                    output = json.loads(output)
                except:
                    print("Failed to parse the response:")
                    print(output)
                    continue
                if verbose:
                    print(f"Prompt: {prompt}\n\nOutput: {output['text']}")
                # 如果没有错误则退出重试循环
                if "error" not in output:
                    request_output = output
                    break
                else:
                    print(f"Failed to process the request: {output['error']}")
                    print(f"Resending the request: {pload}")

        request_end_time = time.perf_counter()
        
        pbar.update(1)        
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=request_output["timestamps"],
            lifetime_events=request_output.get("lifetime_events", None)
        )

async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
    verbose: bool = False,
    burst_count: int = 0  # 新增参数：burst 模式
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, process_name, request_rate, request_cv, burst_count=burst_count
    ):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request.prompt,
                request.prompt_len,
                request.output_len,
                best_of,
                use_beam_search,
                verbose
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)

    # 计算每个请求的端到端时间
    total_request_time = sum(req.end_time - req.start_time for req in request_results)
    avg_request_time = total_request_time / len(request_results)
    print(f"Average end-to-end request time: {avg_request_time:.4f} seconds")

    # 统计其他指标
    ftl_values = [req.ftl for req in request_results]
    tpot_values = [req.tpot for req in request_results]
    total_ftl = sum(ftl_values)
    total_tpot = sum(tpot_values)
    median_ftl = statistics.median(ftl_values)
    median_tpot = statistics.median(tpot_values)
    avg_ftl = total_ftl / len(request_results)
    avg_tpot = total_tpot / len(request_results)

    count_above_0_1 = sum(1 for tpot in tpot_values if tpot > 0.1)
    count_above_0_11 = sum(1 for ftl in ftl_values if ftl > 0.25)
    percentage_above_0_1 = (count_above_0_1 / len(request_results)) * 100
    percentage_above_0_11 = (count_above_0_11 / len(request_results)) * 100

    print(f"Percentage of TFT > 0.25: {percentage_above_0_11:.2f}%")
    print(f"Percentage of Tpot > 0.1: {percentage_above_0_1:.2f}%")
    print(f"Average FTL (First Token Latency): {avg_ftl:.4f} seconds")
    print(f"Average TPOT (Token Per Output Token): {avg_tpot:.4f} seconds")
    print(f"Median FTL: {median_ftl:.4f} seconds")
    print(f"Median TPOT: {median_tpot:.4f} seconds")
    return request_results

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    input_requests = sample_requests(args.dataset, args.num_prompts)
    print("Sampling done. Start benchmarking...")
    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.time()
    request_results = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
            args.verbose,
            burst_count=args.burst_count  # 传递 burst_count 参数
        )
    )
    benchmark_end_time = time.time()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print("Throughput:")
    print(f"\t{args.num_prompts / benchmark_time:.2f} requests/s")
    total_tokens = sum(req.prompt_len + req.output_len for req in input_requests)
    print(f"\t{total_tokens / benchmark_time:.2f} tokens/s")
    total_output_tokens = sum(req.output_len for req in input_requests)
    print(f"\t{total_output_tokens / benchmark_time:.2f} output tokens/s")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="distserve", choices=["distserve", "vllm", "deepspeed"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the (preprocessed) dataset."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts-req-rates", type=str, required=True,
        help='[(num_prompts, request_rate)] or [(num_prompts, request_rate, burst_flag)] where burst_flag: 1 启用 burst 模式（固定 burst 数量），0 不启用。'
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="请求间隔的变异系数。",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: env var EXP_RESULT_ROOT)"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        required=True,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir>"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)"
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    
    if args.exp_result_root is None:
        if "EXP_RESULT_ROOT" not in os.environ:
            print("Error: EXP_RESULT_ROOT is not set in environment variables")
            sys.exit(1)
        args.exp_result_root = os.getenv("EXP_RESULT_ROOT")
        
    if args.exp_result_prefix is None:
        args.exp_result_prefix = args.backend
        
    if args.port is None:
        args.port = BACKEND_TO_PORTS[args.backend]
        
    # 固定 burst 数量，可根据需要调整
    FIXED_BURST_COUNT = 25

    num_prompts_request_rates = eval(args.num_prompts_req_rates)
    for tup in num_prompts_request_rates:
        if len(tup) == 3:
            num_prompts, request_rate, burst_flag = tup
            burst_count = FIXED_BURST_COUNT if burst_flag == 1 else 0
        else:
            num_prompts, request_rate = tup
            burst_count = 0
        print("===================================================================")
        print(f"Running with num_prompts={num_prompts}, request_rate={request_rate}, burst_count={burst_count}")
        args.num_prompts = num_prompts
        args.request_rate = request_rate
        args.burst_count = burst_count
        output_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{args.exp_result_prefix}-{num_prompts}-{request_rate}.exp")
        main(args)
        time.sleep(1)

