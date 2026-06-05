# RAG 问答系统 — 项目报告（工单6.1）

> 项目路径：`D:\Desktop\4-rag-hermes工单6.1\rag-hermes\teams\team\project`
> 文档：招股说明书1（武汉兴图新科）+ 招股说明书2（武汉力源信息）
> 生成日期：2026-06-01

---

## 一、项目结构

```
4-rag-hermes工单6.1/
├── rag-hermes/
│   ├── teams/
│   │   ├── MinerU_out/                          # MinerU PDF解析输出
│   │   │   ├── 招股说明书1-无水印/
│   │   │   │   └── auto/
│   │   │   │       ├── images/                  # 365张解析出的图片
│   │   │   │       ├── 招股说明书1-无水印.md
│   │   │   │       └── 招股说明书1-无水印_content_list.json  # 3817个block
│   │   │   └── 招股说明书2/
│   │   │       └── 招股说明书2/
│   │   │           └── auto/
│   │   │               ├── images/              # 282张解析出的图片
│   │   │               └── 招股说明书2_content_list.json    # 4071个block
│   │   ├── artifacts/evaluation/                # 评测数据（待补充）
│   │   └── team/project/                        # ← 核心项目代码
│   │       ├── src/
│   │       │   ├── api/                         # FastAPI 服务
│   │       │   │   ├── main.py                  # 接口路由
│   │       │   │   ├── schemas.py               # 请求/响应模型
│   │       │   │   └── templates/               # HTML模板
│   │       │   ├── chunker/
│   │       │   │   └── text_splitter.py         # 文档分块（512token）
│   │       │   ├── embedder/
│   │       │   │   └── embed.py                 # bge-m3向量化
│   │       │   ├── generator/
│   │       │   │   └── answer_gen.py            # LLM答案生成
│   │       │   ├── loader/
│   │       │   │   └── mineru_loader.py         # MinerU解析结果加载
│   │       │   ├── memory/
│   │       │   │   └── redis_memory.py          # Redis短期记忆
│   │       │   ├── query/
│   │       │   │   └── understander.py          # 查询理解
│   │       │   ├── retriever/
│   │       │   │   ├── hybrid_retriever.py      # 混合检索（核心）
│   │       │   │   └── reranker.py              # bge-reranker精排
│   │       │   ├── store/
│   │       │   │   ├── milvus_store.py          # Milvus向量库
│   │       │   │   └── keyword_store.py         # BM25关键词索引
│   │       │   └── config.py                    # 全局配置
│   │       ├── scripts/
│   │       │   ├── image_describer.py           # 图片→文字描述（MIMO）
│   │       │   └── ingest_images.py             # 图片描述入库
│   │       ├── data/
│   │       │   ├── source_docs/                 # content_list.json
│   │       │   ├── preview/                     # 分块预览
│   │       │   ├── image_descriptions/          # MIMO生成的描述
│   │       │   └── bm25_index.pkl               # BM25缓存
│   │       ├── logs/rag.log                     # 运行日志
│   │       ├── tests/
│   │       └── requirements.txt
│   ├── rag-project-工单.md                      # 原始工单文档
│   ├── 在线RAG实现方案.md                        # 设计方案
│   ├── FILES.md
│   ├── 招股说明书1-无水印.pdf
│   └── 招股说明书2.pdf
```

---

## 二、功能结构

### 2.1 核心流程

```
用户提问
   ↓
查询理解（指代消解 + 同义词扩展）
   ↓
混合检索（向量 ×0.7 + BM25 ×0.3）
   ├── 向量检索：bge-m3 → Milvus Top-20
   ├── BM25检索：jieba分词 → BM25 Top-20
   └── 合并 → Reranker精排 Top-5
   ↓
答案生成（DeepSeek-v4-pro / MIMO-v2.5-pro）
   ↓
输出答案 + 来源引用
```

### 2.2 各模块职责

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置中心 | `config.py` | 统一管理环境变量、模型路径、API Key |
| 查询理解 | `understander.py` | 指代消解、同义词扩展（营业额→营业收入） |
| 文档加载 | `mineru_loader.py` | 加载 MinerU 解析结果 |
| 文档分块 | `text_splitter.py` | RecursiveCharacterTextSplitter，chunk_size=512 |
| 向量化 | `embed.py` | bge-m3 生成 1024 维向量 |
| 向量存储 | `milvus_store.py` | IvfFlat 索引，Inner Product 距离 |
| 关键词检索 | `keyword_store.py` | BM25Okapi，jieba 分词 |
| 混合检索 | `hybrid_retriever.py` | 融合 + reranker + 多种兜底策略 |
| 精排 | `reranker.py` | bge-reranker 交叉编码器 |
| 答案生成 | `answer_gen.py` | DeepSeek/MIMO + 上下文注入 |
| 短期记忆 | `redis_memory.py` | 最近 N 轮对话，24h TTL |
| API服务 | `main.py` | FastAPI，/query 和 /query/stream |
| 图片描述 | `image_describer.py` | MIMO-v2-omni 多模态看图→文字 |
| 图片入库 | `ingest_images.py` | 图片描述→向量化→Milvus |

### 2.3 数据流

```
PDF → MinerU解析 → content_list.json
  → chunker分块 → preview/chunks.json
  → embedder向量化 → Milvus + BM25缓存
  → 用户查询 → retriever检索 → generator回答
```

---

## 三、优化方案

### 3.1 检索优化

| 优化项 | 文件 | 行数 | 效果 |
|--------|------|:----:|------|
| BM25兜底 | hybrid_retriever.py | +30行 | BM25高分chunk直通reranker |
| 同义词扩展 | hybrid_retriever.py | +15行 | "营业额"→"营业收入" |
| 定义表降权 | hybrid_retriever.py | +8行 | 术语表降权70% |
| 短语兜底 | hybrid_retriever.py | +40行 | 6字以上短语BM25全量搜 |
| 数值兜底 | hybrid_retriever.py | +10行 | query含数值时搜BM25 |
| 关键词对兜底 | hybrid_retriever.py | +30行 | "大客户+销售处"精准配对 |

### 3.2 图片理解（工单 #04）

| 步骤 | 方案 | 说明 |
|------|------|------|
| 图片提取 | MinerU → `images/` | 两张招股书共647张原始图片 |
| 图片筛选 | content_list判断 | 有标题/有来源的13张有意义图片 |
| 图片描述 | MIMO-v2-omni API | 生成结构图/流程图/业务图的文字描述 |
| 入库 | ingester → Milvus | 13条image_description chunk |
| 检索 | 关键词对兜底 | "大客户+销售处"类问题直接BM25配对 |

### 3.3 生成优化

| 优化项 | 改动 | 效果 |
|--------|------|------|
| 推理允许 | pormpt加规则#5 | "仅次于汽车电子"→能推理出"汽车电子第一" |
| 不编造 | 规则#4 部分回答 | 没数据时诚实说"未直接给出" |

### 3.4 性能指标

| 阶段 | 耗时 |
|------|:----:|
| Embedding | ~90ms |
| 向量检索 | ~10-40ms |
| BM25检索 | ~20-40ms |
| Reranker精排(5条) | ~2-3s |
| LLM生成 | ~1-2s |
| **全链路** | **~3-5s** |

---

## 四、遇到的问题与解决方案

### 问题1：RAGAS评分nan

**现象：** 评测时大量指标为 nan，总分算不出来

**根因：** RAGAS 内部让 DeepSeek 做 judge，DeepSeek 不支持 n>1 参数

**解决：** 包装 LLM，强制 n=1

---

### 问题2：对比类问题失败

**现象：** "对比招商银行和平安银行"只检索到平安银行

**根因：** query太长，语义被稀释，单次检索无法覆盖两家公司

**解决：** 查询分解（MapReduce）——按公司名拆子查询再合并

---

### 问题3：模型用自己知识补充

**现象：** Q4（平安营运利润）模型用自己的知识分析"营运利润更能反映持续盈利能力"

**解决：** prompt加约束——"只回答有依据的部分"

---

### 问题4：图片描述错误

**现象：** p309 的 Mouser 仓库图被当作 IC 市场增长图

**根因：** 文本"以下为Mouser仓库内景"在正文中，图片本身就是仓库照片，不是市场图

**结论：** 不是错误，图片和文字匹配。IC市场数据在文本里（p72/p309）

---

### 问题5：检索漏关键chunk

**现象：** Q5（大客户销售处数量）答案说"无法确定"

**根因：** 关键词"大客户+销售处"匹配到的 p113 被 reranker 截断

**解决：** 添加关键词对兜底——query同时含两个词时，不依赖 reranker，直接从BM25捞

---

### 问题6：图片入库后检索不到

**现象：** 13 条 image_description 写入 Milvus 后，查询不命中

**根因：** BM25 缓存没有及时重建，旧缓存不包含新 chunk

**解决：** 删除 bm25_index.pkl，重启 API 自动重建

---

### 问题7：JSON编码错误

**现象：** Windows 下 ingest_images.py 报 UnicodeDecodeError

**根因：** WSL2 生成的是 UTF-8 文件，Windows Python 默认用 GBK

**解决：** `open(..., encoding="utf-8")`

---

## 五、工单完成状态

| 工单 | 状态 | 说明 |
|------|:----:|------|
| #01 基础问答 | ✅ | PDF解析+混合检索+答案生成 |
| #02 优化 | ✅ | BM25兜底+同义词扩展 |
| #03 表格解析 | ✅ | table_semantic+table_json |
| #04 图像解析 | ⚠️ 部分 | 13张图已描述入库，工单16题基本覆盖 |
| #05-#13 | ⬜ | 待开展 |

## 六、已知限制

| 限制 | 说明 |
|------|------|
| Reranker 5条 | 精排候选少，复杂问题可能截断关键数据 |
| BM25需重启 | 新增数据后需删除缓存+重启API |
| 负增长数据缺失 | 招股书2文本未提及IC行业负增长 |
| 部分图片无标题 | 57张图片因无标题/来源被跳过 |
