"""
Embedding 模型微调 — 自定义训练循环
绕开 SentenceTransformerTrainer 的版本兼容问题
"""
import json
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from tqdm import tqdm

# ── 配置 ──
MODEL_PATH = r"D:\Desktop\eleven_project\models\bge-base-zh-v1.5"
TRAIN_DATA = r"D:\Desktop\eleven_project\data\train.jsonl"
OUTPUT_DIR = r"D:\Desktop\eleven_project\output\bge-base-zh-v1.5-finetuned"

EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1


def load_train_data(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            for pos_text in item.get("pos", []):
                examples.append(InputExample(texts=[query, pos_text]))
    return examples


def collate_fn(batch):
    return batch


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    print(f"Loading data: {TRAIN_DATA}")
    train_examples = load_train_data(TRAIN_DATA)
    print(f"Training samples: {len(train_examples)}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps,
    )

    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress:
            texts_a = [ex.texts[0] for ex in batch]
            texts_b = [ex.texts[1] for ex in batch]

            features_a = model.tokenize(texts_a)
            features_b = model.tokenize(texts_b)
            features_a = {k: v.to(device) for k, v in features_a.items()}
            features_b = {k: v.to(device) for k, v in features_b.items()}

            loss = train_loss.forward(
                sentence_features=[features_a, features_b],
                labels=None,
            )

            loss.backward()
            optimizer.step()
            if warmup_steps > 0:
                scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(OUTPUT_DIR)
    print("=" * 50)
    print(f"Training done! Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
