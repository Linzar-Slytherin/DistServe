import argparse
import asyncio
import json
import aiohttp
import time
from typing import List, Dict

# 假设的 FastAPI URL（你需要替换成实际的 URL）
API_URL = "http://localhost:8400/update_resources"

async def send_resource_change_request(session: aiohttp.ClientSession, operation: str, params: list) -> dict:
    """发送资源变动请求"""
    payload = {
        "operation": operation,
        "params": params
    }
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Failed to send resource change request: {response.status}")
                return {}
    except Exception as e:
        print(f"Error sending request: {e}")
        return {}

async def execute_operations(operations: List[Dict], session: aiohttp.ClientSession):
    """根据时间戳执行操作"""
    # 获取程序开始时的时间戳（假设为 0）
    start_time = time.time()

    for operation in operations:
        timestamp = operation["timestamp"]
        operation_type = operation["operation"]
        params = operation["params"]

        # 计算等待时间，直到时间戳到达
        wait_time = start_time + timestamp - time.time()

        if wait_time > 0:
            # 等待直到时间戳到达
            print(f"Waiting for {wait_time:.2f} seconds to execute operation: {operation_type} with params {params}")
            await asyncio.sleep(wait_time)

        print(f"Executing operation: {operation_type} with params {params}")
        result = await send_resource_change_request(session, operation_type, params)
        print(f"Result: {json.dumps(result, indent=4)}")

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Execute operations based on timestamps.")
    parser.add_argument(
        "--operations-file",
        type=str,
        required=True,
        help="Path to the JSON file containing the list of operations, e.g., 'operations.json'.",
    )

    args = parser.parse_args()

    # 从文件读取操作
    try:
        with open(args.operations_file, "r") as file:
            operations = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {args.operations_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {args.operations_file}")
        exit(1)

    # 按时间戳排序操作（确保按照时间顺序执行）
    operations.sort(key=lambda x: x["timestamp"])

    async with aiohttp.ClientSession() as session:
        await execute_operations(operations, session)

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
