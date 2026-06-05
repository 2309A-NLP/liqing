#!/bin/bash
# Embedding 微调 — bge-base-zh-v1.5 (WSL + CUDA)
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/mnt/d/Desktop/eleven_project/models/bge-base-zh-v1.5"
TRAIN_DATA="/mnt/d/Desktop/eleven_project/data/train.jsonl"
OUTPUT_DIR="/mnt/d/Desktop/eleven_project/output/bge-base-zh-v1.5-finetuned"
PYTHON="/home/lqing/miniconda3/envs/ft_env/bin/python"

echo "Model:  $MODEL_PATH"
echo "Data:   $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"

$PYTHON -m torch.distributed.run --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --train_group_size 8 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --fp16 \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 500 \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章："

echo "Training done!"
