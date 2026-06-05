# 这一层专门负责“大模型怎么调用”，把接口细节和业务逻辑分开。
from functools import lru_cache
from typing import Iterator, Tuple

from .config import get_deepseek_model_name, load_llm_client


# 讲解提示：这个函数负责一次性调用大模型并拿回完整答案。
# 它适合“等模型生成完再返回”的普通接口场景。
@lru_cache(maxsize=1)
def call_deepseek(prompt: str, max_tokens: int) -> Tuple[str, str]:
    # 先拿到已经配置好的客户端，后面就可以直接发起请求。
    client = load_llm_client()
    # 模型名从配置里读取，方便之后切换模型而不用改业务代码。
    model = get_deepseek_model_name()
    # 组织消息列表，让模型知道系统身份和用户问题。
    resp = client.chat.completions.create(
        model=model,
        messages=[
            # system 消息用来规定模型整体的说话方式和角色。
            {"role": "system", "content": "你是一个严谨、自然、支持多轮对话的中文助手。"},
            # user 消息里放的是这次真正要回答的内容。
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    # 把模型返回的文本取出来，避免空值导致后续报错。
    answer = resp.choices[0].message.content or ""
    return answer, model


# 讲解提示：这个函数负责流式调用大模型。
# 它适合边生成边展示的场景，前端体验会更好。
def stream_deepseek(prompt: str, max_tokens: int) -> Iterator[Tuple[str, str]]:
    # 同样先准备客户端。
    client = load_llm_client()
    # 同样读取当前配置的模型名。
    model = get_deepseek_model_name()
    # 打开 stream=True 后，模型会持续返回增量内容。
    stream = client.chat.completions.create(
        model=model,
        messages=[
            # system 消息继续负责约束模型行为。
            {"role": "system", "content": "你是一个严谨、自然、支持多轮对话的中文助手。"},
            # user 消息继续承载本轮问题。
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=True,
    )
    # 先发一个空片段，告诉调用方“流已经开始了”。
    yield "", model
    # 后面每个 chunk 都可能只包含一小段新增文本。
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta, model
