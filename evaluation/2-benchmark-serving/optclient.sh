#!/bin/bash
set -e
export USE_DUMMY_WEIGHT=1
echo "Starting distllm client... (for full evaluation, OPT-13B)"
python3 /workspace/DistServe/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /workspace/DATASET/sharegpt.ds --num-prompts-req-rates "[(100,2)]" --exp-result-dir opt-13b-sharegpt
