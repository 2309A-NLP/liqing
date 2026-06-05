#!/bin/bash
# 评估对比 — 基座 vs 微调模型
set -e

export CUDA_VISIBLE_DEVICES=0

BASE_MODEL="/mnt/d/Desktop/eleven_project/models/bge-base-zh-v1.5"
FINETUNED_MODEL="/mnt/d/Desktop/eleven_project/output/bge-base-zh-v1.5-finetuned"
CORPUS_FILE="/mnt/d/Desktop/eleven_project/data/test_corpus.jsonl"
TEST_FILE="/mnt/d/Desktop/eleven_project/data/test_queries.jsonl"
PYTHON="/home/lqing/miniconda3/envs/ft_env/bin/python"

OUTPUT_DIR="/mnt/d/Desktop/eleven_project/output/eval_results"
mkdir -p "$OUTPUT_DIR"

echo "=== Evaluating base model ==="
$PYTHON scripts/evaluate.py \
    --model_name_or_path "$BASE_MODEL" \
    --corpus_file "$CORPUS_FILE" \
    --test_file "$TEST_FILE" \
    --batch_size 256 \
    --output_file "$OUTPUT_DIR/base_model.json"

echo ""
echo "=== Evaluating finetuned model ==="
if [ -d "$FINETUNED_MODEL" ]; then
    $PYTHON scripts/evaluate.py \
        --model_name_or_path "$FINETUNED_MODEL" \
        --corpus_file "$CORPUS_FILE" \
        --test_file "$TEST_FILE" \
        --batch_size 256 \
        --output_file "$OUTPUT_DIR/finetuned_model.json"
else
    echo "Finetuned model not found at $FINETUNED_MODEL"
fi

echo ""
echo "=== Comparison ==="
if [ -f "$OUTPUT_DIR/base_model.json" ] && [ -f "$OUTPUT_DIR/finetuned_model.json" ]; then
    $PYTHON -c "
import json
with open('$OUTPUT_DIR/base_model.json') as f: b = json.load(f)
with open('$OUTPUT_DIR/finetuned_model.json') as f: ft = json.load(f)
print(f'{\"Metric\":<15} {\"Base\":<10} {\"Finetuned\":<10} {\"Delta\":<10}')
print('-' * 45)
for k in b['metrics']:
    bv, fv = b['metrics'][k], ft['metrics'][k]
    delta = fv - bv
    sign = '+' if delta >= 0 else ''
    print(f'{k:<15} {bv:<10.4f} {fv:<10.4f} {sign}{delta:.4f}')
"
fi

echo ""
echo "Done!"
