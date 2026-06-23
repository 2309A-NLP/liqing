#!/usr/bin/env python3
"""
精确定位苗族女孩图中的人脸位置。
输出原图上人脸的精确像素范围。
"""
from PIL import Image
import numpy as np

img = Image.open('/mnt/d/浏览器下载/数字人1.png')
arr = np.array(img)
w, h = arr.shape[1], arr.shape[0]
print(f"原图: {w}x{h}")

# 转 HSV 检测肤色区域
import colorsys

# 直接检查各区域的 RGB，找出真正的肤色（R高，G/B适中）
# 扫描每个 Y 区域，输出 X=700-950 的像素情况
print("\n=== X=700-950 区域逐行分析 ===")
print("Y范围\t| 平均R\t| 平均G\t| 平均B\t| R-G\t| R-B\t| 说明")
print("-" * 70)

for y_start in range(0, 600, 10):
    y_end = y_start + 10
    row = arr[y_start:y_end, 700:950, :]
    r, g, b = row[:,:,0].mean(), row[:,:,1].mean(), row[:,:,2].mean()
    rg = r - g
    rb = r - b
    note = ""
    if r > 180 and r > g and r > b and rg > 5:
        note = "← 肤色"
    elif b > r and b > g and g > 120:
        note = "← 蓝衣/绿植"
    elif r > 230 and g > 230 and b > 230:
        note = "← 白色/高光"
    elif r < 80 and g < 80 and b < 80:
        note = "← 暗部"
    print(f"Y={y_start:3d}-{y_end:3d}\t| R={r:.0f}\t| G={g:.0f}\t| B={b:.0f}\t| {rg:+.0f}\t| {rb:+.0f}\t{note}")

# 更精确地找眼睛/嘴巴
print("\n\n=== 寻找五官（高对比度区域的峰值）===")
for y in range(100, 500, 2):
    row = arr[y, 750:900, :]
    r, g, b = row[:,0].mean(), row[:,1].mean(), row[:,2].mean()
    std = row.std()
    if std > 55:
        print(f"Y={y}: std={std:.1f} R={r:.0f} G={g:.0f} B={b:.0f}")
