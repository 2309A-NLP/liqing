RAG 多角色扮演项目接口说明

一、服务概览

本项目是一个基于 FastAPI、Milvus、Redis、DeepSeek 的多角色 RAG 服务，核心能力包括：

多角色扮演
向量检索
BM25 关键词检索
检索重排
短期会话记忆
长期记忆
生成式问答
二、接口列表

健康检查
接口 GET /health

用途 确认服务是否正常运行。

返回示例 { "status": "ok", "service": "Milvus Hybrid RAG API", "version": "2.1.0" }

聊天问答主接口
接口 POST /chat

用途 发起一次 RAG 问答。

请求示例 { "session_id": "sess_001", "user_id": "user_001", "question": "请介绍一下这个项目的功能", "role_name": "assistant", "top_k": 8, "bm25_top_k": 30, "rerank_top_k": 5, "max_tokens": 1024 }

字段说明

必填字段

session_id 会话 ID，用于区分不同聊天上下文
question 用户问题
可选字段

user_id 用户 ID，默认空字符串
role_name 角色名称，默认 assistant 当前支持：
assistant
lawyer
doctor
top_k dense 检索召回条数
bm25_top_k BM25 召回条数
rerank_top_k 重排后返回条数
max_tokens 大模型最大输出 token 数
三、/chat 返回结构

返回示例 { "session_id": "sess_001", "user_id": "user_001", "role_name": "assistant", "answer": "这是回答内容", "short_memory": [ { "role": "user", "content": "你好", "user_id": "user_001", "role_name": "assistant" }, { "role": "assistant", "content": "你好，有什么可以帮你？", "user_id": "user_001", "role_name": "assistant" } ], "retrieved": [ { "id": "123", "score": 0.91, "source": "chat", "domain": "memory", "title": "长期记忆", "role": "assistant", "keywords": "preference", "content": "用户偏好……", "vector_text": "用户偏好……", "memory_type": "memory", "retrieval_source": "memory", "session_id": "", "user_id": "" } ], "request_id": "8b1d8f2f4d2c4fd6a9d5b7f6f8b2f1a1", "latency_ms": 1284, "retrieval_count": 5, "model_name": "deepseek-chat", "current_role_config": { "role_name": "assistant", "persona": "你是一个严谨的中文RAG助手。", "style": "专业、简洁、准确", "rules": "优先使用检索到的知识和对话记忆回答。" }, "history_turns": 3, "history_messages": 6, "history_user_messages": 3, "history_assistant_messages": 3, "memory_hit": true, "retrieval_stats": { "dense": 8, "bm25": 30, "memory": 4, "merged": 42, "reranked": 5 }, "prompt_preview": "你当前扮演的角色是：你是一个严谨的中文RAG助手。..." }

返回字段说明

基础字段

session_id
user_id
role_name
answer
历史消息

short_memory Redis 中读取的最近历史消息
history_turns 历史轮数
history_messages 历史消息总条数
history_user_messages 用户消息条数
history_assistant_messages 助手消息条数
检索相关

retrieved 最终重排后的结果
retrieval_count 最终返回的检索条数
retrieval_stats 检索统计信息：
dense
bm25
memory
merged
reranked
角色相关

current_role_config 当前角色配置说明
记忆相关

memory_hit 是否命中长期记忆
调试相关

request_id 请求唯一 ID
latency_ms 本次请求耗时
model_name 使用的大模型名称
prompt_preview prompt 预览
四、其他接口

重置会话
接口 POST /session/reset

用途 清空某个 session_id 的短期历史。

请求示例 { "session_id": "sess_001" }

返回示例 { "status": "ok", "session_id": "sess_001" }

获取角色预设
接口 GET /role/presets

用途 获取当前系统支持的角色列表和配置。

返回示例 { "assistant": { "role_name": "assistant", "persona": "你是一个严谨的中文RAG助手。", "style": "专业、简洁、准确", "rules": "优先使用检索到的知识和对话记忆回答。" }, "lawyer": { "role_name": "lawyer", "persona": "你是一名专业法律顾问。", "style": "严谨、条理清晰、避免绝对化表述", "rules": "不得编造法条；不确定时明确提示需要进一步核实。" }, "doctor": { "role_name": "doctor", "persona": "你是一名专业医疗健康顾问。", "style": "谨慎、专业、温和", "rules": "不得替代线下诊疗；涉及危险症状时建议及时就医。" } }

查看长期记忆
接口 GET /memory/list

用途 查看长期记忆记录。

请求参数

limit 默认 50，最大 500
memory_type 可选过滤类型，如：
summary
preference
task
fact
返回示例 { "count": 2, "limit": 50, "memory_type": "", "items": [ { "id": "xxxx", "session_id": "sess_001", "user_id": "user_001", "role_name": "assistant", "memory_type": "preference", "content": "用户喜欢简洁回答", "vector_text": "用户喜欢简洁回答", "source": "chat", "domain": "memory", "title": "长期记忆", "role": "assistant", "keywords": "preference", "importance": 0.8, "created_at": 1710000000 } ] }

五、前端对接建议

前端建议重点展示以下区域：

聊天主区
调试信息区
检索结果区
角色说明区
历史信息区
六、接口特点总结

当前接口已经比较适合前端使用，因为它同时兼顾了：

用户可见回答
检索调试信息
角色扮演信息
记忆命中情况
历史上下文情况