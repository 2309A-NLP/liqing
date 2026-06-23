#!/usr/bin/env python3
"""
完整的 avatar 构建流程：
1. 从视频提取帧
2. 缩小到 576x768（匹配参考avatar尺寸）
3. 在缩小帧上做人脸检测（快且准）
4. 缩放坐标回原图尺寸
5. 在原图上裁 face_imgs
"""
import os, sys, pickle, glob, shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm

VIDEO = "/mnt/d/浏览器下载/苗族数字人形象生成 (1).mp4"
AVATAR_ID = "miaozu_girl_fenghuang_v3"
BASE = "/home/lqing/LiveTalking"
AVATAR_DIR = f"{BASE}/data/avatars/{AVATAR_ID}"
FULL_DIR = f"{AVATAR_DIR}/full_imgs"
FACE_DIR = f"{AVATAR_DIR}/face_imgs"
COORDS_PATH = f"{AVATAR_DIR}/coords.pkl"
IMG_SIZE = 256  # face_imgs 输出尺寸
REF_W, REF_H = 576, 768  # 检测用缩小尺寸

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
print(f"Video: {VIDEO}")
print(f"Avatar: {AVATAR_ID}")

# 清理重建目录
for d in [AVATAR_DIR, FULL_DIR, FACE_DIR]:
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# ── 1. 提取帧 ──
print("\n[1] 提取视频帧...")
cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # OpenCV 读出来是 BGR
    frames.append(frame)
cap.release()
print(f"  提取 {len(frames)} 帧")

# 保存 full_imgs（原尺寸）和检测用缩小帧
orig_h, orig_w = frames[0].shape[:2]
print(f"  原图尺寸: {orig_w}x{orig_h}")

small_frames = []  # 缩小到 576x768 用于检测
for i, f in enumerate(tqdm(frames, desc="Saving frames")):
    # 保存原尺寸 full_imgs
    cv2.imwrite(f"{FULL_DIR}/{i:08d}.png", f)
    # 缩小用于检测
    small = cv2.resize(f, (REF_W, REF_H))
    small_frames.append(small)

# ── 2. 人脸检测（在缩小的帧上跑）──
print(f"\n[2] 人脸检测 (在 {REF_W}x{REF_H} 上)...")
sys.path.insert(0, BASE)
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
            detected = sum(1 for d in dets if d is not None)
            total_batches = (len(small_frames) + batch_size - 1) // batch_size
            print(f"  batch {i//batch_size+1}/{total_batches}: {detected}/{len(dets)} faces", end='\r')
    except RuntimeError as e:
        if batch_size == 1:
            raise
        batch_size //= 2
        print(f"\n  OOM, reducing batch to {batch_size}")
        continue
    break

print()
detected_count = sum(1 for p in predictions if p is not None)
print(f"  检测结果: {detected_count}/{len(predictions)} 帧有脸")

if detected_count < len(predictions) * 0.8:
    print(f"\n  ⚠️ 检测率只有 {detected_count/len(predictions)*100:.0f}%，可能不准")
    print(f"  使用固定人脸坐标作为后备")

# ── 3. 计算人脸框 ──
# face_detection 返回 [x1, y1, x2, y2] 在缩小帧上
# 需要缩放到原图
scale_x = orig_w / REF_W  # 720/576 = 1.25
scale_y = orig_h / REF_H  # 1280/768 = 1.6667

pads = [0, 10, 0, 0]
pady1, pady2, padx1, padx2 = pads
results = []

for rect, image in zip(predictions, frames):
    if rect is None:
        # 没有检测到脸 — 使用固定位置（从参考帧分析得出）
        rect = [256, 340, 448, 570]  # 上一轮已验证正确的坐标
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
print(f"\n[3] 坐标范围统计:")
print(f"  x: {boxes[:,0].min():.0f}-{boxes[:,2].max():.0f}")
print(f"  y: {boxes[:,1].min():.0f}-{boxes[:,3].max():.0f}")

# 时序平滑
T = 5
for i in range(len(boxes)):
    if i + T > len(boxes):
        window = boxes[len(boxes) - T:]
    else:
        window = boxes[i:i + T]
    boxes[i] = np.mean(window, axis=0)

# ── 4. 生成 face_imgs + coords ──
print(f"\n[4] 生成 face_imgs ({IMG_SIZE}x{IMG_SIZE})...")
coord_list = []
for idx, (rect, frame) in enumerate(zip(boxes, frames)):
    y1, y2, x1, x2 = [int(v) for v in rect]
    face_frame = frame[y1:y2, x1:x2]
    resized = cv2.resize(face_frame, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(f"{FACE_DIR}/{idx:08d}.png", resized)
    coord_list.append((y1, y2, x1, x2))

with open(COORDS_PATH, 'wb') as f:
    pickle.dump(coord_list, f)

# ── 5. 验证 ──
# 保存验证图
check = frames[0].copy()
y1, y2, x1, x2 = coord_list[0]
cv2.rectangle(check, (x1, y1), (x2, y2), (0, 0, 255), 3)
cv2.imwrite(f"{AVATAR_DIR}/check_face_rect.png", check)

# 输出统计
full_files = sorted(os.listdir(FULL_DIR))
face_files = sorted(os.listdir(FACE_DIR))
print(f"\n✅ 完成! {AVATAR_ID}")
print(f"   full_imgs: {len(full_files)} 帧, {orig_w}x{orig_h}")
print(f"   face_imgs: {len(face_files)} 帧, {IMG_SIZE}x{IMG_SIZE}")
print(f"   coords[0]: {coord_list[0]}")
print(f"   人脸大小: {x2-x1}x{y2-y1}")
print(f"\n启动命令:")
print(f"   python app.py --transport webrtc --model wav2lip --avatar_id {AVATAR_ID}")
