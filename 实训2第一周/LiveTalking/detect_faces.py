#!/usr/bin/env python3
"""对已提取的 full_imgs 运行人脸检测（先缩小再检测），生成 face_imgs + coords.pkl"""
import os, sys, pickle, glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

AVATAR_ID = "miaozu_girl_fenghuang_v2"
BASE = "/home/lqing/LiveTalking"
AVATAR_DIR = f"{BASE}/data/avatars/{AVATAR_ID}"
FULL_DIR = f"{AVATAR_DIR}/full_imgs"
FACE_DIR = f"{AVATAR_DIR}/face_imgs"
COORDS_PATH = f"{AVATAR_DIR}/coords.pkl"
IMG_SIZE = 256  # face_imgs 输出尺寸

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# 确保 face_imgs 目录存在
os.makedirs(FACE_DIR, exist_ok=True)

# 读取 full_imgs
input_img_list = sorted(glob.glob(os.path.join(FULL_DIR, '*.[jpJP][pnPN]*[gG]')))
print(f"Loaded {len(input_img_list)} frames")

# 读取帧
frames = []
for p in tqdm(input_img_list, desc="Reading frames"):
    frame = cv2.imread(p)
    frames.append(frame)

# 缩小到 360x640 用于人脸检测
scale_w, scale_h = 360, 640
small_frames = []
for f in tqdm(frames, desc="Resizing for detection"):
    small = cv2.resize(f, (scale_w, scale_h))
    small_frames.append(small)

print("Running face detection...")
from avatars.wav2lip import face_detection
detector = face_detection.FaceAlignment(
    face_detection.LandmarksType._2D, flip_input=False, device=device)

batch_size = 16
predictions = []
while True:
    try:
        for i in range(0, len(small_frames), batch_size):
            batch = np.array(small_frames[i:i+batch_size])
            dets = detector.get_detections_for_batch(batch)
            predictions.extend(dets)
            print(f"  batch {i//batch_size+1}/{(len(small_frames)+batch_size-1)//batch_size}: detected {sum(1 for d in dets if d is not None)}/{len(dets)} faces")
    except RuntimeError as e:
        if batch_size == 1:
            raise
        batch_size //= 2
        print(f"OOM, reducing batch to {batch_size}")
        continue
    break

# 检测统计
detected_count = sum(1 for p in predictions if p is not None)
print(f"\nFace detection: {detected_count}/{len(predictions)} frames have faces")

# 生成 face_imgs + coords
# 对于缩小后的检测框，需要缩放回原图坐标
orig_h, orig_w = frames[0].shape[:2]
scale_x = orig_w / scale_w   # 720/360 = 2
scale_y = orig_h / scale_h   # 1280/640 = 2

pads = [0, 10, 0, 0]
pady1, pady2, padx1, padx2 = pads
results = []

for rect, image in zip(predictions, frames):
    if rect is None:
        rect = [0, 0, image.shape[1], image.shape[0]]
    else:
        # 缩放回原图坐标
        rect = [int(rect[0] * scale_x), int(rect[1] * scale_y),
                int(rect[2] * scale_x), int(rect[3] * scale_y)]
    
    y1 = max(0, rect[1] - pady1)
    y2 = min(image.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(image.shape[1], rect[2] + padx2)
    results.append([x1, y1, x2, y2])

boxes = np.array(results)

# 时序平滑
T = 5
for i in range(len(boxes)):
    if i + T > len(boxes):
        window = boxes[len(boxes) - T:]
    else:
        window = boxes[i:i + T]
    boxes[i] = np.mean(window, axis=0)

# 保存 face_imgs + coords
coord_list = []
for idx, (rect, frame) in enumerate(zip(boxes, frames)):
    y1, y2, x1, x2 = [int(v) for v in rect]
    face_frame = frame[y1:y2, x1:x2]
    resized = cv2.resize(face_frame, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(f"{FACE_DIR}/{idx:08d}.png", resized)
    coord_list.append((y1, y2, x1, x2))

with open(COORDS_PATH, 'wb') as f:
    pickle.dump(coord_list, f)

# 验证
face_files = sorted(os.listdir(FACE_DIR))
print(f"\n✅ 完成！")
print(f"   full_imgs: {len(input_img_list)} 帧")
print(f"   face_imgs: {len(face_files)} 帧")
print(f"   coords[0]: {coord_list[0]}")
print(f"   原图尺寸: {orig_w}x{orig_h}")
