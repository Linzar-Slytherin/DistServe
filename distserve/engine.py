import time
import copy
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import asyncio
import math
import argparse

import ray
from ray.util.placement_group import PlacementGroup

from distserve.config import (
    ModelConfig, 
    DisaggParallelConfig, 
    ParallelConfig, 
    CacheConfig, 
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.logger import init_logger
from distserve.request import (
    SamplingParams,
    Request,
    create_request,
)
from distserve.tokenizer import get_tokenizer
from distserve.utils import Counter
from distserve.single_stage_engine import (
    StepOutput,
    ContextStageLLMEngine,
    DecodingStageLLMEngine
)
from distserve.lifetime import LifetimeEvent, LifetimeEventType

logger = init_logger(__name__)



class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig,
        num_context_engines: int = 2,
        num_decoding_engines: int = 2,
    ):
        # 校验数量在 1～8 之间
        assert 1 <= num_context_engines <= 8, "context engine 数量必须在 1~8 之间"
        assert 1 <= num_decoding_engines <= 8, "decoding engine 数量必须在 1~8 之间"
        self.num_context_engines = num_context_engines
        self.num_decoding_engines = num_decoding_engines

        self.model_config = model_config
        self.disagg_parallel_config = disagg_parallel_config
        self.cache_config = cache_config
        self.context_sched_config = context_sched_config
        self.decoding_sched_config = decoding_sched_config
        self.request_counter = Counter()
        self.count = 1
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        # 用于 context 阶段生产的中间结果（比如 kv cache 或 token 输出）
        self.bridge_queue = asyncio.Queue()
        # 为 decode 阶段单独维护一个队列列表，每个 decode engine 一个队列
        self.decode_queues = [asyncio.Queue() for _ in range(num_decoding_engines)]

        # 初始化 placement groups（和之前类似）
        logger.info("Initializing placement group")
        placement_groups = self._init_placement_groups()

        # 动态创建 context 引擎列表
        self.context_engines: List[ContextStageLLMEngine] = []
        for i in range(num_context_engines):
            engine = ContextStageLLMEngine(
                cengine_id=i + 1,
                bridge_queue=self.bridge_queue,  # 多个 context 引擎共享同一队列
                model_config=model_config,
                parallel_config=disagg_parallel_config.context,
                cache_config=cache_config,
                sched_config=context_sched_config,
                placement_groups=placement_groups,
                engine_on_new_step_output_callback=self._on_new_step_output_callback,
                engine_on_new_lifetime_event_callback=self._on_new_lifetime_event_callback,
            )
            logger.info(f"Initializing context stage LLM engine {i + 1}")
            self.context_engines.append(engine)

        # 动态创建 decode 引擎列表，每个 decode 引擎有自己的队列
        self.decoding_engines: List[DecodingStageLLMEngine] = []
        for i in range(num_decoding_engines):
            engine = DecodingStageLLMEngine(
                dengine_id=i + 1,
                # 将 decode 阶段的输入队列设为各自专用的队列
                bridge_queue=self.decode_queues[i],
                model_config=model_config,
                parallel_config=disagg_parallel_config.decoding,
                cache_config=cache_config,
                sched_config=decoding_sched_config,
                placement_groups=placement_groups,
                # 下面传入 context 引擎的 kv cache 清理回调需要传入所有 context engine 的回调函数
                clear_migrated_blocks_callback=self.context_engines[0].clear_migrated_blocks_callback,  # 假设所有 context engine 实现一致
                engine_on_new_step_output_callback=self._on_new_step_output_callback,
                engine_on_new_lifetime_event_callback=self._on_new_lifetime_event_callback,
                clear_migrated_blocks_callback2=(self.context_engines[1].clear_migrated_blocks_callback if len(self.context_engines) >= 2 else None)
            )
            logger.info(f"Initializing decoding stage LLM engine {i + 1}")
            self.decoding_engines.append(engine)

        # 存放请求 id 对应的最终输出（来自 on_new_step_output_callback）
        self.request_outputs: Dict[int, asyncio.Queue[StepOutput]] = {}
        # 存放请求的生命周期事件，调用 generate() 时创建，调用者负责清理
        self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}

        self.engine_initialized = False

    # （保留原有的 _on_new_step_output_callback 与 _on_new_lifetime_event_callback）
    def _on_new_step_output_callback(self, request_id: int, step_output: StepOutput):
        self.request_outputs[request_id].put_nowait(step_output)

    def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent, dont_add_if_dup: bool = False):
        if dont_add_if_dup and \
           len(self.request_lifetime_events[request_id]) > 0 and \
           self.request_lifetime_events[request_id][-1].event_type == event.event_type:
            return
        self.request_lifetime_events[request_id].append(event)

    def _init_placement_groups(self) -> Optional[List[PlacementGroup]]:
        workers_per_placement_group=4
        num_placement_groups=1
        # Create placement groups
        placement_groups = []
        for i in range(num_placement_groups):
            placement_group = ray.util.placement_group(
                [ { "GPU": 1 }] * workers_per_placement_group,
                strategy="STRICT_PACK",
            )
            ray.get(placement_group.ready(), timeout=1000)
            placement_groups.append(placement_group)
        
        return placement_groups
    async def initialize(self):
        # 并发初始化所有的 context 和 decoding 引擎
        init_tasks = [
            engine.initialize()
            for engine in self.context_engines + self.decoding_engines
        ]
        await asyncio.gather(*init_tasks)

        # 对于每个 decode 引擎，依次注册所有 context 引擎的 kvcache memory handles
        registration_tasks = []
        for decode_engine in self.decoding_engines:
            for context_engine in self.context_engines:
                registration_tasks.append(
                    decode_engine.register_kvcache_mem_handles(
                        context_engine.parallel_config,
                        context_engine.kv_cache_mem_handles,
                        context_engine.cengine_id
                    )
                )
        await asyncio.gather(*registration_tasks)

        self.engine_initialized = True

    async def _decode_dispatcher(self):
        """从 context 阶段的 bridge_queue 中取出数据，
           按轮询的方式分发到各个 decode 队列中。"""
        idx = 0
        while True:
            step_output = await self.bridge_queue.get()
            # 将获取到的数据送入当前 decode 队列
            await self.decode_queues[idx].put(step_output)
            idx = (idx + 1) % self.num_decoding_engines
        
    def abort_request(self, request_id: int):
        for engine in self.context_engines:
            engine.abort_request(request_id)
        for engine in self.decoding_engines:
            engine.abort_request(request_id)

    def _remote_call_all_workers(self, func_name: str, *args):
        handlers = []
        for engine in self.context_engines:
            handlers.extend(engine._remote_call_all_workers_async(func_name, *args))
        for engine in self.decoding_engines:
            handlers.extend(engine._remote_call_all_workers_async(func_name, *args))
        return ray.get(handlers)

    def _remote_call_all_workers_async(self, func_name: str, *args):
        handlers = []
        for engine in self.context_engines:
            handlers.extend(engine._remote_call_all_workers_async(func_name, *args))
        for engine in self.decoding_engines:
            handlers.extend(engine._remote_call_all_workers_async(func_name, *args))
        return handlers


    async def _start_my_event_loop(self):
        pass
    
    async def start_all_event_loops(self):
        logger.info("Starting LLMEngine event loops")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."

        # 初始化所有引擎的事件循环任务
        tasks = []
        for engine in self.context_engines:
            tasks.append(engine.start_event_loop())
        for engine in self.decoding_engines:
            tasks.append(engine.start_event_loop())
        # 加入 dispatcher 任务
        tasks.append(self._decode_dispatcher())
        await asyncio.gather(*tasks)

    async def generate(
        self,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ) -> AsyncGenerator[StepOutput, None]:
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."

        # 轮询选取一个 context engine
        selected_engine = self.context_engines[self.count % self.num_context_engines]
        cengine_id = selected_engine.cengine_id
        self.count += 1

        req = create_request(
            prompt,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
            cengine_id
        )
        self.request_outputs[req.request_id] = asyncio.Queue()
        self.request_lifetime_events[req.request_id] = []

        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Issued))
        selected_engine.add_request(req)

        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break

        del self.request_outputs[req.request_id]



        
def add_engine_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--context-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=16)
    
    parser.add_argument("--context-sched-policy", type=str, default="fcfs")
    parser.add_argument("--context-max-batch-size", type=int, default=256)
    parser.add_argument("--context-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    
    parser.add_argument("--simulator-mode", action="store_true")
    parser.add_argument("--profiler-data-path", type=str, default=None)
    parser.add_argument("--gpu-mem-size-gb", type=float, default=None)
    
