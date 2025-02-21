from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import asyncio
import time
import json
from threading import Lock
from queue import Queue
import argparse
import json
from typing import AsyncGenerator, List, Tuple
import asyncio
import time
import traceback
import sys, os
import signal

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

import distserve
import distserve.engine
from distserve.llm import AsyncLLM
from distserve.request import SamplingParams
from distserve.utils import random_uuid, set_random_seed
from distserve.logger import init_logger
from distserve.single_stage_engine import StepOutput
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.lifetime import json_encode_lifetime_events

import ray

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

# 创建锁来确保 `update_resources` 先执行
update_lock = Lock()
# 使用队列来排队资源更新请求
update_queue = Queue()
# 使用队列来排队 `generate` 请求
generate_queue = Queue()

# 锁用于确保 `generate` 请求在资源更新完成后执行
generate_lock = Lock()

@app.post("/update_resources")
async def update_resources(request: Request) -> JSONResponse:
    """
    该路由接收资源更新请求，并将操作放入队列等待执行。
    请求体格式应为：
    {
        "operation": "change",  # 操作类型（"remove" 或 "add" 或 "change"）
        "params": ["context", 0, "decode", 0]  # 参数（操作的资源信息）
    }
    """
    try:
        # 获取请求体中的数据
        request_data = await request.json()
        operation = request_data.get("operation")
        params = request_data.get("params")

        # 检查参数有效性
        if not operation or not params:
            return JSONResponse(status_code=400, content={"message": "Invalid request parameters"})

        # 将请求放入队列
        update_queue.put((operation, params))

        # 执行 `update_resources` 操作
        await process_update_resources()

        return JSONResponse(status_code=200, content={"message": f"Request to {operation} with params {params} queued"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

async def process_update_resources():
    """确保先处理 `update_resources` 请求"""
    while not update_queue.empty():
        operation, params = update_queue.get()
        with update_lock:
            # 执行资源更新操作
            if operation == "remove":
                await engine.engine.remove(params[0])
            elif operation == "add":
                await engine.engine.add_engine(*params)
            elif operation == "change":
                if len(params) == 4:
                    await engine.engine.change_engine(*params)
                else:
                    print(f"Invalid params for 'change' operation: {params}")
            else:
                print(f"Unknown operation: {operation}")
        update_queue.task_done()
        print(f"Completed {operation} with params: {params}")

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request."""
    logger.info("Received a request.")
    
    # 将请求放入生成队列，等待 `update_resources` 完成
    generate_queue.put(request)
    
    # 等待直到所有资源更新操作完成
    await wait_for_update_resources()

    # 处理生成请求
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(
        request_id, prompt=prompt, sampling_params=sampling_params
    )

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for step_output in results_generator:
                text_output = step_output.request.get_response()
                ret = {"text": text_output}
                yield (json.dumps(ret)).encode("utf-8")

        async def abort_request() -> None:
            await engine.abort(request_id)

        background_tasks = BackgroundTasks()
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_outputs: List[Tuple[StepOutput, float]] = []  # (step_output, timestamp)
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_outputs.append((step_output, time.perf_counter()))

        request_events = engine.get_and_pop_request_lifetime_events(request_id)
        text_output = prompt + ''.join([step_output[0].new_token for step_output in final_outputs])
        ret = {
            "text": text_output,
            "timestamps": [step_output[1] for step_output in final_outputs],
            "lifetime_events": json_encode_lifetime_events(request_events)
        }
        return JSONResponse(ret)

async def wait_for_update_resources():
    """等待直到资源更新完成"""
    # 如果没有 `generate` 请求等待，允许继续执行资源更新操作
    if generate_queue.empty():
        await process_update_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    distserve.engine.add_engine_cli_args(parser)
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    ray.init()
    
    engine = AsyncLLM.from_engine_args(args)

    uvicorn_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    async def main_coroutine():
        task2 = asyncio.create_task(uvicorn_server.serve())
        
        async def start_event_loop_wrapper():
            try:
                task = asyncio.create_task(engine.start_event_loop())
                await task
            except Exception as e:
                traceback.print_exc()
                task2.cancel()
                os._exit(1)  # Kill myself, or it will print tons of errors. Don't know why.
        
        task1 = asyncio.create_task(start_event_loop_wrapper())
        
        try:
            await task2
        except:
            pass
    
    asyncio.run(main_coroutine())
