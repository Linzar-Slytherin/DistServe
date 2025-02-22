import asyncio
import math
import json  # 导入 json 模块，用于格式化输出


# 模拟 llm.engine 接口（实际使用时请替换为真实接口）
class Engine:
    async def remove(self, gpu_id):
        print(f"await llm.engine.remove({gpu_id})")

    async def add(self, actor, gpu_id):
        print(f"await llm.engine.add({actor},{gpu_id})")

    async def change_engine(self, original_gpu, original_role, new_gpu, new_role):
        # 参数格式：原来的 gpu id、原来的角色、新的 gpu id（同一值）和新的角色
        print(f"await llm.engine.change_engine({original_gpu}, {original_role!r}, {new_gpu}, {new_role!r})")


class DummyLLM:
    def __init__(self):
        self.engine = Engine()


llm = DummyLLM()


class GPUResourceManager:
    def __init__(self, n, a, start_time=0):
        self.n = n
        self.a = a

        # 初始化每个 GPU 的累计生存时间，单位自定义，初始均为 0
        self.gpu_lifetime = {i: 0 for i in range(n)}
        self.gpu_role = {}
        self.prefill_pool = []  # 存储当前分配到 prefill 池的 GPU id
        self.decode_pool = []  # 存储当前分配到 decode 池的 GPU id
        self.active_gpus = set(range(n))  # 初始所有 GPU 均活跃

        init_prefill = round((self.a / (self.a + 1)) * n)
        for gpu in range(n):
            if gpu < init_prefill:
                self.prefill_pool.append(gpu)
                self.gpu_role[gpu] = "prefill"
            else:
                self.decode_pool.append(gpu)
                self.gpu_role[gpu] = "decode"

        self.last_update_time = start_time
        self.interface_ops = []
    async def adjust_pools_2(self):
        # 当前池中的数量
        p = len(self.prefill_pool)
        d = len(self.decode_pool)
        # 复活待用 GPU 列表
        recovered = [gpu for gpu in self.active_gpus
                     if gpu not in self.prefill_pool and gpu not in self.decode_pool
                     and self.gpu_lifetime[gpu] == 0]
        # 当前 active 数
        T = p + d + len(recovered)
        if T == 0 or self.a <= 0:
            return

        # 计算当前流水线吞吐量
        current_tp = min(d, p / self.a)
        # 初始化可转移数
        pt = 0
        dt = 0
        if current_tp > 0:
            # 判断当前瓶颈
            if current_tp == d:
                p_min = math.ceil(self.a * d)
                pt = max(p - p_min, 0)
                dt = 0
            else:
                d_min = math.ceil(p / self.a)
                dt = max(d - d_min, 0)
                pt = 0
        else:
            pt = p
            dt = d
        # 计算最优分配：达到最大流水线吞吐量时 active 数T的理想分配
        if T < 2:
            return
        # 枚举可能的 decode 数 d（1 ≤ d < T），选择使流水线吞吐最大化的方案
        best_tp = -1
        best_d = None
        for d in range(1, T):
            current_tp = min(d, (T - d) / self.a)
            if current_tp > best_tp:
                best_tp = current_tp
                best_d = d
        d_max = best_d
        p_max = T - best_d
        # ===== 根据可转移情况调整 =====
        # 规则1：如果 pt == 0 且 dt == 0，则仅用 recovered GPU补足各池到最优
        if pt == 0 and dt == 0 and current_tp!=0:
            while self.prefill_pool.__len__() < p_max and recovered:
                candidate = recovered.pop(0)
                self.prefill_pool.append(candidate)
                if self.gpu_role[candidate] != "prefill":
                    self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "prefill"))
                    self.gpu_role[candidate] = "prefill"
                else:
                    self.interface_ops.append(("add",  self.gpu_role[candidate],candidate))
            while self.decode_pool.__len__() < d_max and recovered:
                candidate = recovered.pop(0)
                self.decode_pool.append(candidate)
                if self.gpu_role[candidate] != "decode":
                    self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "decode"))
                    self.gpu_role[candidate] = "decode"
                else:
                    self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
        # 规则2：如果 pt > 0 且 dt == 0  (即当前瓶颈为 decode)
        elif pt > 0 and dt == 0 and current_tp!=0:
            # 2a. 如果当前 p < p_max，则先利用 recovered GPU补足缺口
            if p < p_max:
                while len(self.prefill_pool) < p_max and recovered:
                    candidate = recovered.pop(0)
                    self.prefill_pool.append(candidate)
                    if self.gpu_role[candidate] != "prefill":
                        self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "prefill"))
                        self.gpu_role[candidate] = "prefill"
                    else:
                        self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
                while len(self.decode_pool) < d_max and recovered:
                    candidate = recovered.pop(0)
                    self.decode_pool.append(candidate)
                    if self.gpu_role[candidate] != "decode":
                        self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "decode"))
                        self.gpu_role[candidate] = "decode"
                    else:
                        self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
            # 2b. 如果 p == p_max，则直接用 recovered 补足 decode 到 d_max
            elif p == p_max:
                while len(self.decode_pool) < d_max and recovered:
                    candidate = recovered.pop(0)
                    self.decode_pool.append(candidate)
                    if self.gpu_role[candidate] != "decode":
                        self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "decode"))
                        self.gpu_role[candidate] = "decode"
                    else:
                        self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
            # 2c. 如果 p > p_max, 则 surplus = p - p_max，转移 prefill 中“最年轻”的 surplus 个 GPU到 decode
            else:
                surplus = p - p_max
                transfer_count = 0
                while surplus > 0 and transfer_count < surplus:
                    # “最年轻”：生存时间最小
                    candidate = min(self.prefill_pool, key=lambda gpu: self.gpu_lifetime[gpu])
                    self.prefill_pool.remove(candidate)
                    self.decode_pool.append(candidate)
                    self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
                    self.gpu_role[candidate] = "decode"
                    transfer_count += 1
        # 规则3：如果 dt > 0 且 pt == 0 (即当前瓶颈为 prefill)
        elif dt > 0 and pt == 0 and current_tp!=0:
            transfer_count = 0
            # 设计循环：只要 prefill 数小于 p_max且转移次数小于 dt，从 decode_pool中选择“最年长”的 GPU转入 prefill
            while len(self.prefill_pool) < p_max and transfer_count < dt and self.decode_pool:
                # “最年长”：生存时间最大
                candidate = max(self.decode_pool, key=lambda gpu: self.gpu_lifetime[gpu])
                self.decode_pool.remove(candidate)
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "decode", candidate, "prefill"))
                self.gpu_role[candidate] = "prefill"
                transfer_count += 1
            # 循环结束后，用 recovered GPU补足两池到最优
            while len(self.prefill_pool) < p_max and recovered:
                candidate = recovered.pop(0)
                self.prefill_pool.append(candidate)
                if self.gpu_role[candidate] != "prefill":
                    self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "prefill"))
                    self.gpu_role[candidate] = "prefill"
                else:
                    self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
            while len(self.decode_pool) < d_max and recovered:
                candidate = recovered.pop(0)
                self.decode_pool.append(candidate)
                if self.gpu_role[candidate] != "decode":
                    self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "decode"))
                    self.gpu_role[candidate] = "decode"
                else:
                    self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
        # 规则4：如果当前吞吐量为0，则所有 active GPU 均可变
        else:
            print("maxxxxxxxxxxxxxxxxxxx")
            # current_tp == 0：吞吐量为0
            # 计算理想最优分配：p_max = ceil((a/(a+1))*T)，d_max = T - p_max
            p_max = math.ceil((self.a / (self.a + 1)) * T)
            d_max = T - p_max
            # 对所有 active_gpus 按生存时间从短到长排序
            sorted_gpus = sorted(list(self.active_gpus), key=lambda gpu: self.gpu_lifetime[gpu])
            # 分配：生存时间较长的分配给 prefill，生存时间较短的分配给 decode
            # 注意：sorted_gpus[0] 为生存时间最短，sorted_gpus[-1] 为生存时间最长
            # 因此，取后 p_max 个分配给 prefill，取前 d_max 个分配给 decode
            prefill_candidates = sorted_gpus[-p_max:]
            decode_candidates = sorted_gpus[:d_max]
            # 先确保这些 GPU不在其它池中，再分别分配
            for gpu in prefill_candidates:
                if gpu in self.decode_pool:
                    self.decode_pool.remove(gpu)
                if gpu not in self.prefill_pool:
                    self.prefill_pool.append(gpu)
                if self.gpu_role[gpu] != "prefill":
                    self.interface_ops.append(("change_engine", gpu, self.gpu_role[gpu], gpu, "prefill"))
                    self.gpu_role[gpu] = "prefill"
            for gpu in decode_candidates:
                if gpu in self.prefill_pool:
                    self.prefill_pool.remove(gpu)
                if gpu not in self.decode_pool:
                    self.decode_pool.append(gpu)
                if self.gpu_role[gpu] != "decode":
                    self.interface_ops.append(("change_engine", gpu, self.gpu_role[gpu], gpu, "decode"))
                    self.gpu_role[gpu] = "decode"

    async def adjust_pools(self):
        """
        调整逻辑：
          1. 设 T = 当前活跃 GPU 数。
          2. 枚举 d 从 1 到 T-1，计算 throughput = min(d, (T-d)/a)，选择 throughput 最大时对应的 d，
             得到目标分配：
               target_decode = d*, target_prefill = T - d*
          3. 根据目标分配调整当前 prefill_pool 与 decode_pool：
             - 如果 prefill_pool 数不足，则依次使用恢复 GPU（优先原角色为 prefill，否则转换）
               或从 decode_pool转换来补充。
             - 如果 prefill_pool 数过多，则逐个从 prefill_pool转换到 decode_pool。
             - 类似地，若 decode_pool 数不足，则补充恢复 GPU或转换现有 GPU。
          4. 最后，将所有剩余（恢复且未分配）GPU归入 decode_pool，
             对于这些 GPU，若原角色为 decode则直接 add，否则用 change_engine 转换为 decode。
        """
        T = len(self.active_gpus)
        if T < 2:
            return

        # 枚举可能的 decode 数 d（1 ≤ d < T），选择使流水线吞吐最大化的方案
        best_tp = -1
        best_d = None
        for d in range(1, T):
            current_tp = min(d, (T - d) / self.a)
            if current_tp > best_tp:
                best_tp = current_tp
                best_d = d
        target_decode = best_d
        target_prefill = T - best_d

        # 找出所有恢复（生存时间==0）且未分配到任何池中的 GPU
        recovered = [gpu for gpu in self.active_gpus
                     if gpu not in self.prefill_pool and gpu not in self.decode_pool
                     and self.gpu_lifetime[gpu] == 0]

        # 调整 prefill_pool 至目标 target_prefill
        while len(self.prefill_pool) < target_prefill:
            candidate = None
            # 1. 尝试从恢复 GPU中选择原角色为 prefill的
            for gpu in recovered:
                if self.gpu_role[gpu] == "prefill":
                    candidate = gpu
                    break
            if candidate is not None:
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
                recovered.remove(candidate)
                continue
            # 2. 尝试选择恢复 GPU中原角色为 decode（需转换）
            for gpu in recovered:
                if self.gpu_role[gpu] == "decode":
                    candidate = gpu
                    break
            if candidate is not None:
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "decode", candidate, "prefill"))
                self.gpu_role[candidate] = "prefill"
                recovered.remove(candidate)
                continue
            # 3. 如恢复 GPU不足，则从 decode_pool中转换一个 GPU
            if self.decode_pool:
                candidate = self.decode_pool.pop(0)
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "decode", candidate, "prefill"))
                self.gpu_role[candidate] = "prefill"
            else:
                break

        while len(self.prefill_pool) > target_prefill:
            candidate = self.prefill_pool.pop()
            self.decode_pool.append(candidate)
            self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
            self.gpu_role[candidate] = "decode"

        # 调整 decode_pool 至目标 target_decode
        while len(self.decode_pool) < target_decode:
            candidate = None
            # 1. 尝试选择恢复 GPU中原角色为 decode
            for gpu in recovered:
                if self.gpu_role[gpu] == "decode":
                    candidate = gpu
                    break
            if candidate is not None:
                self.decode_pool.append(candidate)
                self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
                recovered.remove(candidate)
                continue
            # 2. 尝试选择恢复 GPU中原角色为 prefill（需转换）
            for gpu in recovered:
                if self.gpu_role[gpu] == "prefill":
                    candidate = gpu
                    break
            if candidate is not None:
                self.decode_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
                self.gpu_role[candidate] = "decode"
                recovered.remove(candidate)
                continue
            # 3. 如恢复 GPU不足，则从 prefill_pool中转换一个 GPU
            if self.prefill_pool:
                candidate = self.prefill_pool.pop(0)
                self.decode_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
                self.gpu_role[candidate] = "decode"
            else:
                break

        # 将所有剩余恢复 GPU全部放入 decode_pool
        for candidate in recovered:
            self.decode_pool.append(candidate)
            if self.gpu_role[candidate] == "decode":
                self.interface_ops.append(("add", self.gpu_role[candidate],candidate))
            else:
                self.interface_ops.append(("change_engine", candidate, self.gpu_role[candidate], candidate, "decode"))
                self.gpu_role[candidate] = "decode"

    async def process_event(self, event):
        current_time = event.get("time", self.last_update_time)
        delta = current_time - self.last_update_time

        # 更新所有活跃 GPU（生存时间>=0）的累计生存时间
        for gpu in self.active_gpus:
            if self.gpu_lifetime[gpu] >= 0:
                self.gpu_lifetime[gpu] += delta
        self.last_update_time = current_time

        # --- Step 1：处理抢占 ---
        for gpu in event.get("preempted", []):
            if gpu in self.active_gpus:
                self.gpu_lifetime[gpu] = -1
                self.active_gpus.remove(gpu)
                if gpu in self.prefill_pool:
                    self.prefill_pool.remove(gpu)
                if gpu in self.decode_pool:
                    self.decode_pool.remove(gpu)
                self.interface_ops.append(("remove", gpu))

        # --- Step 2：处理恢复 ---
        for gpu in event.get("resumed", []):
            if gpu not in self.active_gpus:
                self.active_gpus.add(gpu)
                self.gpu_lifetime[gpu] = 0

        # --- Step 3：调整池分配以最大化流水线吞吐 ---
        await self.adjust_pools_2()

        # 统一执行本时间戳内所有接口操作
        await self.execute_interface_ops()
        self.interface_ops.clear()

    async def execute_interface_ops(self):
        """
        依次执行本时间戳内待发出的所有接口调用，并将每个操作单独写入文件
        """
        with open('tranceop.json', 'a') as logfile:
            for op in self.interface_ops:
                log_entry = {}
                if op[0] == "remove":
                    gpu = op[1]
                    await llm.engine.remove(gpu)
                    log_entry = {"operation": "remove", "params": [gpu]}
                elif op[0] == "add":
                    gpu = op[2]
                    actor = op[1]
                    await llm.engine.add(actor, gpu)
                    log_entry = {"operation": "add", "params": [actor, gpu]}
                elif op[0] == "change_engine":
                    gpu, from_role, new_gpu, to_role = op[1], op[2], op[3], op[4]
                    await llm.engine.remove(op[1])
                    await llm.engine.change_engine(from_role, gpu, to_role, new_gpu)
                    log_entry = {"operation": "change", "params": [from_role, gpu, to_role, new_gpu]}
                else:
                    print("Unknown op:", op)
                # 将每个操作记录单独写入文件
                json.dump(log_entry, logfile)
                logfile.write(",\n")


    def print_status(self):
        print("【GPU 生存时间】", self.gpu_lifetime)
        print("【活跃 GPU】", self.active_gpus)
        print("【prefill 池】", self.prefill_pool)
        print("【decode 池】", self.decode_pool)
        print("【GPU 角色】", self.gpu_role)
        print("=====================================")


# 示例主程序
async def main():
    n = 8
    a = 2.1  # a 表示 decode 与 prefill 吞吐量之比：1 个 decode 满负荷需要 1.9 个 prefill
    manager = GPUResourceManager(n, a)

    # 模拟事件序列：
    events = [
        {
            "time": 10,
            "preempted": [4, 5, 6],
            "resumed": [],
            "request_speed": 100
        },
        {
            "time": 20,
            "preempted": [1, 2],
            "resumed": [4, 5, 6],
            "request_speed": 150
        }
    ]

    for event in events:
        print(f"处理时间戳 {event['time']} 的事件...")
        await manager.process_event(event)
        manager.print_status()


if __name__ == "__main__":
    asyncio.run(main())
