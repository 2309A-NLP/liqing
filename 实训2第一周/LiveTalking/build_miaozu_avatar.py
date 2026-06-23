#!/usr/bin/env python3
"""
从静态图构建苗族女孩数字人 avatar。
流程：
  1. 裁原始图为半身像（头+肩+上胸，576x768）
  2. 在裁剪图中定位人脸区域
  3. 生成 550 帧 full_imgs（全部同一张半身像）
  4. 裁人脸并缩放到 256x256 → face_imgs
  5. 写 coords.pkl
"""

import os
import pickle
import shutil
from PIL import Image, ImageDraw

AVATAR_ID = "miaozu_girl_fenghuang"
SRC = "/mnt/d/浏览器下载/数字人1.png"
BASE = os.path.dirname(os.path.abspath(__file__))
AVATAR_DIR = os.path.join(BASE, "data", "avatars", AVATAR_ID)
FULL_DIR = os.path.join(AVATAR_DIR, "full_imgs")
FACE_DIR = os.path.join(AVATAR_DIR, "face_imgs")
COORDS_PATH = os.path.join(AVATAR_DIR, "coords.pkl")
NUM_FRAMES = 550
FACE_SIZE = 256  # img_size for wav2lip

def main():
    print("=" * 50)
    print(f"构建 avatar: {AVATAR_ID}")
    print(f"源图: {SRC}")
    print(f"帧数: {NUM_FRAMES}")
    print(f"人脸尺寸: {FACE_SIZE}x{FACE_SIZE}")
    print("=" * 50)

    # ── 1. 清空重建目录 ──
    for d in [AVATAR_DIR, FULL_DIR, FACE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # ── 2. 打开原图 ──
    img = Image.open(SRC)
    w, h = img.size
    print(f"\n[1] 原图尺寸: {w}x{h}")

    # ── 3. 裁半身像 ──
    # 原图中人物位置（像素分析确认）:
    #   银饰头冠:   X=790-1000 / Y=0-350
    #   人脸:       X=830-1020 / Y=370-510
    #   上身(蓝衣): X=550-1200 / Y=500-700
    # 裁图: 包含银饰头冠+人脸+肩膀+上半身
    # 人物中心在原图约 X=1020，居中在 800px 裁图里 X=400
    crop_x1 = 650
    crop_y1 = 280
    crop_x2 = 1450
    crop_y2 = 850
    bust = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    bust_w, bust_h = bust.size
    print(f"[2] 半身像裁图: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}) = {bust_w}x{bust_h}")

    # ── 4. 确定裁剪图中的人脸坐标 ──
    # 原图人脸: (900,410)-(1200,650) — 含额头到下巴
    # 减去裁图偏移量
    face_x1 = 900 - crop_x1
    face_y1 = 410 - crop_y1
    face_x2 = 1200 - crop_x1
    face_y2 = 650 - crop_y1

    print(f"[3] 裁剪图中人脸区域: ({face_x1},{face_y1})-({face_x2},{face_y2})")
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1
    print(f"    人脸尺寸: {face_w}x{face_h}")

    # ── 5. 裁人脸并缩放 ──
    face_region = bust.crop((face_x1, face_y1, face_x2, face_y2))
    face_resized = face_region.resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
    print(f"[4] 人脸裁图 → {FACE_SIZE}x{FACE_SIZE}")

    # ── 6. 生成帧 ──
    print(f"\n[5] 生成 {NUM_FRAMES} 帧...")
    for i in range(NUM_FRAMES):
        # full_imgs: 保存半身像 RGB
        bust_rgb = bust.convert("RGB")
        bust_rgb.save(os.path.join(FULL_DIR, f"{i:08d}.png"), "PNG")

        # face_imgs: 保存缩放到 256x256 的人脸
        face_resized.save(os.path.join(FACE_DIR, f"{i:08d}.png"), "PNG")

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i+1}/{NUM_FRAMES} 帧")

    # ── 7. 写坐标 ──
    # coords.pkl 格式: [(y1, y2, x1, x2), ...]
    # 注意: y 是行(x), x 是列(y)
    coord_list = [(face_y1, face_y2, face_x1, face_x2)] * NUM_FRAMES
    with open(COORDS_PATH, "wb") as f:
        pickle.dump(coord_list, f)
    print(f"\n[6] 坐标已写入: {COORDS_PATH}")
    print(f"    坐标格式: [y1, y2, x1, x2] = [{face_y1}, {face_y2}, {face_x1}, {face_x2}]")

    # ── 8. 验证 ──
    full_files = sorted(os.listdir(FULL_DIR))
    face_files = sorted(os.listdir(FACE_DIR))
    with open(COORDS_PATH, "rb") as f:
        coords = pickle.load(f)

    print(f"\\n[7] 验证:")
    print(f"    full_imgs: {len(full_files)} 帧")
    print(f"    face_imgs: {len(face_files)} 帧")
    print(f"    coords.pkl: {len(coords)} 条坐标")
    full_img = Image.open(os.path.join(FULL_DIR, full_files[0]))
    face_img = Image.open(os.path.join(FACE_DIR, face_files[0]))
    print(f"    full_imgs[0] 尺寸: {full_img.size}")
    print(f"    face_imgs[0] 尺寸: {face_img.size}")
    print(f"    coords[0]: {coords[0]}")
    print(f"\\n✅ Avatar 构建完成！")

    # ── 9. 保存标注验证图 ──
    full_rgb = full_img.convert("RGB")
    draw = ImageDraw.Draw(full_rgb)
    y1, y2, x1, x2 = coords[0]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    check_path = os.path.join(AVATAR_DIR, "check_face_rect.png")
    full_rgb.save(check_path)
    print(f"[8] 验证图已保存: {check_path}")
    print(f"    红框内是 wav2lip 将贴口型的人脸区域")

    # ── 9. 显示图像信息 ──
    # 在终端显示 ASCII 预览（可选信息）
    print(f"\n{'='*50}")
    print(f"启动方式:")
    print(f"  python app.py --model wav2lip --avatar_id {AVATAR_ID} \\")
    print(f"    --transport webrtc --batch_size 16 --modelres 192")

if __name__ == "__main__":
    main()
