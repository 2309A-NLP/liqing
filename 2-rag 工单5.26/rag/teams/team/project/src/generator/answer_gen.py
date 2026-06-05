"""
答案生成模块 — deepseek-v4-pro
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import json
import logging
from typing import List, Dict, Any
import httpx
from src.config import config

logger = logging.getLogger(__name__)


class Generator:
    """RAG 答案生成器"""

    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = config.DEEPSEEK_BASE_URL
        self.model = config.DEEPSEEK_MODEL

    def _build_messages(self, question: str, context_chunks: List[Dict[str, Any]],
                        history: List[Dict[str, Any]] | None = None) -> List[Dict[str, str]]:
        """构建消息列表（提取为公共方法，供流式和非流式共用）"""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            page = chunk.get("page_no", "?")
            source = chunk.get("source_file", "招股说明书")
            context_parts.append(
                f"[{i+1}] (来源: {source} 第{page}页)\n{chunk['text']}"
            )
        context_text = "\n\n".join(context_parts)

        # 自动检测用户语言，支持中英文
        lang_hint = "中文" if any("\u4e00" <= c <= "\u9fff" for c in question) else "English"
        system_prompt = (
            f"你是一个专业的文档问答助手。请基于提供的检索内容回答用户问题。\n\n"
            f"回答要求：\n"
            f"1. 严格基于检索内容回答，不要编造信息\n"
            f"2. 如果检索内容不足以回答问题，明确回答'根据现有资料无法回答该问题'\n"
            f"3. 标注引用来源的页码，格式：[来源: 招股说明书 第X页]\n"
            f"4. 如果涉及数字数据，确保与原文一致\n"
            f"5. 用{lang_hint}回答"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-config.REDIS_HISTORY_N:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        user_prompt = f"检索到的相关内容：\n{context_text}\n\n用户问题：{question}"
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _sources_from_chunks(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从检索结果中提取引用来源"""
        sources = []
        seen = set()
        for chunk in context_chunks:
            key = chunk.get("page_no", 0)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page_no": chunk.get("page_no"),
                    "text": chunk["text"][:200],
                    "score": chunk.get("score", 0),
                })
        return sources

    def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        messages = self._build_messages(question, context_chunks, history)
        answer = self._call_api(messages)
        sources = self._sources_from_chunks(context_chunks)
        return {"answer": answer, "sources": sources}

    def generate_stream(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, Any]] | None = None,
    ):
        """流式生成答案，逐块 yield (chunk_type, data)

        chunks:
          ("source", json)  → 引用来源信息
          ("token", str)    → 一个字或词
          ("done", None)    → 结束
        """
        messages = self._build_messages(question, context_chunks, history)
        sources = self._sources_from_chunks(context_chunks)
        yield ("source", sources)

        # 流式调用 API
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=60) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or line.startswith(":") or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield ("token", content)
        except Exception as e:
            logger.error(f"流式 API 异常: {e}")
            yield ("token", f"[生成失败: {e}]")

        yield ("done", None)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用 deepseek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # 低温度，偏向确定性回答
            "max_tokens": 2048,
        }

        try:
            # 构造完整 URL，兼容 base_url 中带或不带 /v1 的情况
            base = self.base_url.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            url = f"{base}/chat/completions"

            with httpx.Client(timeout=30) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"API 调用失败: {e.response.status_code} {e.response.text}")
            return f"API 调用失败，请检查 API Key 和网络连接。"
        except Exception as e:
            logger.error(f"API 调用异常: {e}")
            return f"生成回答时出现异常: {str(e)}"
