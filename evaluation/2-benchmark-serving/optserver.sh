#!/bin/bash
set -e
export USE_DUMMY_WEIGHT=1
echo "Starting distllm server... (for full evaluation, OPT-13B)"
python3 /workspace/DistServe/evaluation/2-benchmark-serving/2-start-api-server.py --backend distserve --model facebook/opt-13b
