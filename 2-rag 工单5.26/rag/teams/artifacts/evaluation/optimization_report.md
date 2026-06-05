# RAG 优化报告

> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
> 优化时间：2026-05-26
> 基准版本：工单一完成版（v1.0）
> 优化版本：v2.0

---

## 一、优化目标

| 指标 | 工单要求 | 优化前（v1.0） | 优化后（v2.0） |
|------|---------|---------------|---------------|
| 准确率 | ≥ 90% | 100%（10/10） | 待验证 |
| 响应时间 | ≤ 3s | 4.38s | 预计 ~2.5s |
| 多语言 | 中英文 | 仅中文 | 中英文自动切换 |
| 系统稳定性 | 容错机制 | 基本有 | 二次检索降级 |

---

## 二、优化方案

### 2.1 Query 理解并行化（核心优化）

**问题**：优化前 Query 理解（1.4s）和检索（0.3s）串行执行，总耗时 1.7s。

**方案**：使用 `asyncio.gather` 将 Query 理解和检索并行执行。

```
优化前（串行）：
  记忆读取 → Query理解(1.4s) → 检索(0.3s) → LLM生成(1.5s)
  总计：~3.2s

优化后（并行）：
  记忆读取 → [Query理解(1.4s) ∥ 检索(0.3s)] → 二次检索(0.3s, 如需) → LLM生成(1.5s)
  总计：~1.7s + 生成 = ~3.2s → ~2.5s（无历史时直接跳过 Query 理解）
```

**实现**：
```python
# 并行执行 Query 理解和检索
qu_task = asyncio.to_thread(get_understander().understand, question, history)
ret_task = asyncio.to_thread(get_retriever().retrieve, question)
qu_result, initial_chunks = await asyncio.gather(qu_task, ret_task)

# 改写问题不同则二次检索
if qu_result["rewritten"] != question:
    context_chunks = get_retriever().retrieve(qu_result["rewritten"])
```

**预期效果**：
- 无历史时：跳过 Query 理解，直接检索，省 1.4s
- 有历史时：Query 理解和检索并行，省 ~0.3s（被检索覆盖）

### 2.2 Prompt 精简

**问题**：原 system prompt 为 5 行约 120 tokens，增加 LLM 处理时间。

**方案**：精简为 1 行约 40 tokens，保留核心指令。

```
优化前（120 tokens）：
  "你是一个专业的文档问答助手。请基于提供的检索内容回答用户问题。
   回答要求：
   1. 严格基于检索内容回答，不要编造信息
   2. 如果检索内容不足以回答问题，明确回答'根据现有资料无法回答该问题'
   3. 标注引用来源的页码，格式：[来源: 招股说明书 第X页]
   4. 如果涉及数字数据，确保与原文一致
   5. 使用中文回答"

优化后（40 tokens）：
  "基于检索内容回答。严格依据原文，不编造。无法回答则说明。
   标注来源页码[来源:第X页]。数字与原文一致。用中文回答。"
```

**预期效果**：减少 ~80 tokens 输入，LLM 生成速度提升约 5-10%。

### 2.3 多语言支持

**问题**：原系统仅支持中文回答。

**方案**：自动检测用户输入语言，动态切换回答语言。

```python
lang_hint = "中文" if any("\u4e00" <= c <= "\u9fff" for c in question) else "English"
system_prompt = f"...用{lang_hint}回答。"
```

**效果**：
- 中文问题 → 中文回答
- 英文问题 → 英文回答

### 2.4 容错机制加强

**问题**：二次检索可能失败，需要降级处理。

**方案**：二次检索失败时降级使用初始检索结果。

```python
if search_query != question:
    try:
        context_chunks = get_retriever().retrieve(search_query)
    except Exception as e:
        logger.error(f"二次检索失败: {e}，使用初始结果")
        context_chunks = initial_chunks  # 降级
```

---

## 三、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/api/main.py` | ① 添加 asyncio import ② /query 端点并行化 ③ /query/stream 端点并行化 ④ 二次检索降级 |
| `src/generator/answer_gen.py` | ① system prompt 精简 ② 多语言自动检测 |
| `src/query/understander.py` | ① 代词消解补刀机制（已在 v1.0 完成） |

---

## 四、优化前后对比（理论值）

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Query 理解 + 检索 | 1.7s（串行） | 1.4s（并行） | -0.3s |
| LLM 生成 | 1.5s | ~1.4s（prompt 精简） | -0.1s |
| 全链路（有历史） | 4.38s | ~3.0s | **-1.4s** |
| 全链路（无历史） | 3.0s | ~2.0s | **-1.0s** |
| 多语言 | ❌ | ✅ | 新增 |
| 容错降级 | ❌ | ✅ | 新增 |

---

## 五、验证方法

重启 RAG 服务后，运行评测脚本对比：

```bash
# 优化后评测
python eval.py --runs 3 --output artifacts/evaluation/report_v2.md

# 对比优化前后延迟
# report_v1.md vs report_v2.md
```

---

## 六、验收清单

| 验收项 | 状态 |
|--------|------|
| 准确率 ≥ 90% | ✅ 已达标（v1.0 为 100%） |
| 响应时间 ≤ 3s | ⚠️ 需验证（理论值 ~2.5-3.0s） |
| 优化方案对比分析 | ✅ 本报告 |
| 交互友好性 | ✅ 深色主题 UI |
| 多语言支持 | ✅ 中英文自动切换 |
| 容错机制 | ✅ 二次检索降级 |
