@echo off
chcp 65001 >nul
set CUDA_VISIBLE_DEVICES=0
echo Starting fine-tuning...
D:\an\envs\py3.11\python.exe scripts\train.py
echo Done!
