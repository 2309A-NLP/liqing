# 在线 RAG 实现方案

## 1. 目标

当前项目已经完成：
- PDF 离线解析与分块
- 向量化与 Milvus 入库
- FastAPI 基础接口
- Redis 短期记忆接入

下一步的目标是把 `/api/ask` 从“占位问答”升级为真正的在线 RAG：

1. 接收用户问题
2. 生成查询向量
3. 检索 Milvus Top-K chunk
4. 拼接检索上下文
5. 结合短期记忆构造 prompt
6. 调用 LLM 生成回答
7. 返回答案与来源

---

## 2. 当前状态

### 已完成的基础能力
- `Embedder`：可生成文本向量
- `MilvusStore.search()`：可按向量检索
- `AnswerService`：可调用 LLM 生成回答
- `MemoryService`：可管理 Redis 短期记忆
- `/api/ask`：已有接口骨架

### 已完成的离线进展
- 已成功完成离线建库
- 当前知识库中已写入 **683 条 chunk / 向量记录**
- 针对 `招股说明书1-无水印.pdf` 已完成一轮离线优化：
  - 目录页污染显著下降
  - `section_path` 层级基本可用
  - 纯标题 chunk 已明显减少
- 当前离线质量已经达到：**可以进入在线检索验证阶段**

### 当前缺口
- `VectorRetriever` 还是空实现
- `/api/ask` 还没有真正调用检索
- 没有把召回结果转成 `sources`
- 没有把召回文本拼装成 `context`
- 页面还不能展示真实检索结果与来源

---

## 3. 当前阶段的策略调整

原先计划是：

```text
先持续优化 chunk -> 再做在线
```

现在离线已经达到可用水平，因此建议正式切换为：

```text
先做 /api/retrieve 检索验证 -> 再做 /api/ask 完整在线 RAG
```

也就是说，当前重点不再是继续大改 chunker，而是先验证：
- 这 683 条向量记录能否召回正确内容
- 页码、章节、文本片段是否足够支撑问答

---

## 4. 推荐实现顺序

建议按 **B -> A** 的思路推进：

### 阶段 B：先做“检索可视化验证”
先不急着做完整问答，先确认：
- 给一个问题，Milvus 能否召回正确 chunk
- 返回的页码、章节、文本是否合理
- 召回结果是否会被封面、声明页等弱相关 chunk 干扰

这样做的好处：
- 可以先验证离线数据质量
- 更容易定位问题到底出在“检索”还是“生成”
- 便于调试 top_k 和 chunk 质量

### 阶段 A：再做“完整在线 RAG 问答”
在检索验证通过后，再接入 LLM。

---

## 5. 阶段 B：检索可视化验证方案

### 5.1 要实现什么
新增一个只负责“检索结果展示”的接口：
- `POST /api/retrieve`

请求：
```json
{
  "question": "公司主营业务是什么？"
}
```

返回：
- top_k chunk
- chunk_id
- page_start / page_end
- section_path
- text_preview
- score

### 5.2 需要改哪些模块

#### `app/retrieval/vector_retriever.py`
实现：
- 使用 `Embedder` 对问题做向量化
- 调用 `MilvusStore.search()`
- 返回原始召回结果
- 对返回结果做统一结构整理

#### `app/core/schemas.py`
新增检索请求 / 响应模型，例如：
- `RetrieveRequest`
- `RetrieveResponse`

#### `app/api/routes.py`
新增：
- `/api/retrieve`

#### `app/web/templates/index.html`
可选增强：
- 支持调试查看检索结果
- 展示来源页码和章节路径

### 5.3 当前阶段的验证重点
实现 `/api/retrieve` 后，先人工验证几个问题：
- 公司主营业务是什么？
- 控股股东是谁？
- 注册资本是多少？
- 风险因素有哪些？
- 募投项目是什么？

### 5.4 预期收益
- 先确认“库里这 683 条数据能否检索对”
- 为后续接 LLM 打基础
- 识别是否还需要继续优化 chunk 质量

---

## 6. 阶段 A：完整在线 RAG 问答方案

### 6.1 主流程

`POST /api/ask` 的流程建议为：

```text
用户问题
  -> 读取短期记忆
  -> 向量检索 top_k
  -> 构造 context
  -> 调用 LLM 生成回答
  -> 写入本轮 user/assistant 到 Redis
  -> 返回 answer + sources + timing
```

### 6.2 上下文构造

建议上下文由两部分组成：

#### A. 短期记忆上下文
来自 Redis：
- 最近若干轮 user / assistant 对话

作用：
- 解决指代问题
- 保持多轮会话连续性

#### B. 检索证据上下文
来自 Milvus top_k：
- chunk 文本
- 页码
- 章节路径

作用：
- 提供事实依据
- 保证回答有证据支撑

### 6.3 当前阶段的注意点
由于当前知识库前几页仍保留少量封面 / 声明类内容，因此在 `/api/ask` 接入时应注意：
- 优先根据 score 和正文密度筛选上下文
- 尽量不要把明显弱相关的封面块拼入最终 prompt

---

## 7. 推荐模块设计

### 7.1 `VectorRetriever`
职责：
- 把问题转向量
- 从 Milvus 检索
- 统一输出检索结果结构

建议输出字段：
- `chunk_id`
- `doc_id`
- `page_start`
- `page_end`
- `section_path`
- `chunk_index`
- `text`
- `score`

### 7.2 `AnswerService`
职责：
- 接收 `question + context + memory_context`
- 调用 prompt_builder
- 调用 LLMClient

当前基础已经有了，只需要接上真实 context。

### 7.3 `routes.py`
职责：
- 协调 API 层逻辑
- 统计 `retrieve_ms / generate_ms`
- 组装返回 `sources`
- 写 Redis 短期记忆

---

## 8. sources 返回设计

`/api/ask` 最终应该返回：

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "招股说明书1-无水印-080",
      "page_start": 37,
      "page_end": 37,
      "section_path": ["第五节 发行人基本情况", "一、发行人的基本情况"],
      "score": 0.87,
      "text_preview": "公司名称：武汉兴图新科电子股份有限公司..."
    }
  ],
  "retrieve_ms": 42,
  "generate_ms": 913
}
```

### 这样设计的价值
- 可追溯
- 便于前端展示
- 便于评测与调试

---

## 9. 推荐实施节奏

### 第一步：先做检索验证接口
实现：
- `VectorRetriever`
- `/api/retrieve`
- 检索响应模型

### 第二步：验证检索质量
人工验证：
- 召回的 chunk 是否命中正文
- `section_path` 是否合理
- 页码是否可信
- score 是否有区分度

### 第三步：再把 `/api/ask` 接上检索
实现：
- 先检索
- 再构造 context
- 再调用 LLM
- 返回 sources

### 第四步：最后做前端增强
让页面支持：
- 查看来源
- 展示页码
- 展示章节路径
- 展示检索耗时 / 生成耗时

---

## 10. 当前阶段不建议立即做的事情

### 10.1 不先做 Hybrid / BM25 / Reranker
原因：
- 现在连最基础的向量检索链路还没接通
- 先跑通最小闭环更重要

### 10.2 不先做复杂的 prompt 工程
原因：
- 先看基础检索质量
- 后续再按效果调 prompt

### 10.3 不先做长期记忆召回融合
原因：
- 当前只启用 Redis 短期记忆
- 长期记忆留到下一阶段更合适

### 10.4 不继续大改离线 parser / chunker
原因：
- 当前离线质量已达到可用线
- 继续优化边际收益下降
- 先进入在线验证更高效

---

## 11. 推荐技术路线结论

对当前项目，建议按下面顺序推进：

```text
1. 已完成离线优化并重建知识库
2. 现在先做 /api/retrieve 检索验证
3. 再把 /api/ask 接成完整在线 RAG
4. 最后再考虑 BM25 / reranker / 长期记忆
```

---

## 12. 当前执行建议

### 推荐方案（当前版本）
1. 保持当前离线版本不再大改
2. 实现 `/api/retrieve`
3. 用真实问题验证 Milvus 检索效果
4. 通过后接 `/api/ask`
5. 最后做前端来源展示增强

### 我的建议
> **当前最佳下一步是：直接开始实现 `/api/retrieve`，先验证 683 条离线知识能不能稳定召回正确正文。**
