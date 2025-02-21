#!/bin/bash
. /workspace/DistServe/evaluation/ae-scripts/e2e/common.sh

echo "Starting distllm client... (for full evaluation, OPT-13B)"
python3 /workspace/DistServe/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /workspace/DATASET/sharegpt.ds --num-prompts-req-rates "[(300, 0.75), (300, 1.5), (300, 3), (300, 4.5), (300, 6), (300, 6.75), (300, 7.5), (300, 9)]" --exp-result-dir opt-13b-sharegpt
