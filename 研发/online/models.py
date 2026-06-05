# 这一层定义系统里会反复用到的数据结构。
# 把请求、响应、检索结果、角色配置都标准化，后面传递起来更稳定。
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .config import DEFAULT_BM25_TOP_K, DEFAULT_MAX_TOKENS, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K


# 单条检索结果的统一结构。
# 不管是向量召回、BM25 召回还是长期记忆召回，最后都统一成这个格式。
@dataclass
class RetrievedDoc:
    id: str
    score: float
    source: str
    domain: str
    title: str
    role: str
    keywords: str
    content: str
    vector_text: str
    memory_type: str = "knowledge"
    retrieval_source: str = "dense"
    session_id: str = ""
    user_id: str = ""


# 前端发来的聊天请求。
# 这里会校验参数是否合法，避免把错误输入直接传进后端流程。
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(default="", description="用户ID")
    question: str = Field(..., description="用户问题")
    role_name: str = Field(default="assistant", description="角色名称")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    bm25_top_k: int = Field(default=DEFAULT_BM25_TOP_K, ge=1, le=200)
    rerank_top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=50)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)


# 记录一次请求里各类检索的数量，方便观察系统效果。
class RetrievalStats(BaseModel):
    dense: int = 0
    bm25: int = 0
    memory: int = 0
    merged: int = 0
    reranked: int = 0


# 后端返回给前端的聊天响应。
# 除了答案本身，还会把检索、记忆和耗时信息一起返回，便于调试和展示。
class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    role_name: str
    answer: str
    short_memory: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    request_id: str = ""
    latency_ms: int = 0
    retrieval_count: int = 0
    model_name: str = ""
    current_role_config: Dict[str, Any] = Field(default_factory=dict)
    history_turns: int = 0
    history_messages: int = 0
    history_user_messages: int = 0
    history_assistant_messages: int = 0
    memory_hit: bool = False
    retrieval_stats: RetrievalStats = Field(default_factory=RetrievalStats)
    prompt_preview: str = ""
    timings_ms: Dict[str, float] = Field(default_factory=dict)


# 用来表示“这一段内容值不值得存成长期记忆”。
class MemoryCandidate(BaseModel):
    memory_type: str = Field(default="summary")
    memory_text: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    should_save: bool = True


# 角色配置模板。
# 不同角色只是“说话方式”和“规则”不同，底层流程是一样的。
class RoleConfig(BaseModel):
    role_name: str
    persona: str
    style: str = "专业、简洁、准确"
    rules: str = "只依据检索资料和历史记忆回答，不确定时明确说明资料不足。"


ROLE_PRESETS: Dict[str, RoleConfig] = {
    "assistant": RoleConfig(
        role_name="assistant",
        persona="你是一个严谨的中文RAG助手。",
        style="专业、简洁、准确",
        rules="优先使用检索到的知识和对话记忆回答。",
    ),
    "lawyer": RoleConfig(
        role_name="lawyer",
        persona="你是一名专业法律顾问。",
        style="严谨、条理清晰、避免绝对化表述",
        rules="不得编造法条；不确定时明确提示需要进一步核实。",
    ),
    "doctor": RoleConfig(
        role_name="doctor",
        persona="你是一名专业医疗健康顾问。",
        style="谨慎、专业、温和",
        rules="不得替代线下诊疗；涉及危险症状时建议及时就医。",
    ),
}


def role_presets_dict() -> Dict[str, Dict[str, Any]]:
    return {name: cfg.model_dump() for name, cfg in ROLE_PRESETS.items()}


def get_role_config(role_name: str) -> RoleConfig:
    return ROLE_PRESETS.get(role_name, ROLE_PRESETS["assistant"])
