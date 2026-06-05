"""
RAG 交互式查询测试
支持传统RAG和LightRAG双引擎切换
"""

import asyncio
import logging
import aiohttp
import json
import sys

# Windows Proactor事件循环的ConnectionResetError修复
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

API_BASE = "http://localhost:8004"


async def interactive():
    print("=" * 60)
    print("RAG 交互式查询")
    print("输入问题开始查询，输入 q 退出")
    print("命令：")
    print("  engine:auto         - 智能选择引擎（默认）")
    print("  engine:traditional  - 切换到传统RAG")
    print("  engine:lightrag     - 切换到LightRAG")
    print("  mode:local/global/mix - 切换LightRAG查询模式")
    print("=" * 60)

    current_engine = "auto"
    current_mode = "mix"
    session_id = "test_session"

    timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时，LightRAG查询慢
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            print()
            user_input = input("请输入问题: ").strip()
            if not user_input or user_input.lower() == "q":
                break

            # 切换引擎
            if user_input.startswith("engine:"):
                new_engine = user_input.split(":")[1].strip()
                if new_engine in ["auto", "traditional", "lightrag"]:
                    current_engine = new_engine
                    print(f"已切换到 {current_engine} 引擎")
                else:
                    print("无效引擎，可选: traditional/lightrag")
                continue

            # 切换模式
            if user_input.startswith("mode:"):
                new_mode = user_input.split(":")[1].strip()
                if new_mode in ["local", "global", "hybrid", "mix", "naive"]:
                    current_mode = new_mode
                    print(f"已切换到 {current_mode} 模式")
                else:
                    print("无效模式，可选: local/global/hybrid/mix/naive")
                continue

            print(f"\n问题: {user_input}")
            print(f"引擎: {current_engine}")
            if current_engine == "lightrag":
                print(f"模式: {current_mode}")
            print("-" * 40)

            # 构建请求
            payload = {
                "question": user_input,
                "session_id": session_id,
                "engine": current_engine,
                "lightrag_mode": current_mode,
            }

            try:
                async with session.post(f"{API_BASE}/query", json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(result["answer"])
                    else:
                        error = await resp.text()
                        print(f"查询失败: {error}")
            except (aiohttp.ClientError, ConnectionResetError, asyncio.TimeoutError) as e:
                print(f"连接异常: {e}")
                print("服务可能繁忙，请重试")
            except Exception as e:
                print(f"查询出错: {e}")

    print("\n退出查询。")


if __name__ == "__main__":
    asyncio.run(interactive())
