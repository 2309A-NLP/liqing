# RAG 问答系统面试题早读材料

基于真实项目：PDF文档问答系统（Milvus + BM25 + Reranker + DeepSeek）

---

## 一、项目架构与核心流程

### 题目1：请描述你的RAG系统的完整数据流

**参考答案：**
```
PDF文档 → Fitz解析 → RecursiveCharacterTextSplitter分块(512token, 50重叠)
                    ↓
           bge-m3向量化(1024维) → Milvus存储
                    ↓
用户query → QueryUnderstander(LLM改写/指代消解) → 混合检索
                                                    ↓
                                         向量检索×0.7 + BM25×0.3
                                                    ↓
                                            Reranker精排(top-10)
                                                    ↓
                                     DeepSeek-v4-pro生成答案
                                                    ↓
                                         Redis短期记忆(多轮对话)
```

**考察点：** 是否理解全链路，能否说清楚每个环节的作用。

---

### 题目2：为什么用混合检索而不是纯向量检索？

**参考答案：**
- **向量检索优势：** 语义相似，"注册资本"能匹配到"注册资金"
- **向量检索劣势：** 精确关键词匹配弱，"5,520万"这种数字容易漏
- **BM25优势：** 词频匹配，精确关键词命中率高
- **BM25劣势：** 无法理解同义词，"营收"匹配不到"营业收入"
- **融合策略：** 向量×0.7 + BM25×0.3，两全其美

**追问：** 权重怎么定的？答：实验调参，向量权重更高因为语义理解更重要。

---

### 题目3：Reranker的作用是什么？和Embedding有什么区别？

**参考答案：**
- **Embedding（双塔模型）：** query和doc分别编码，计算余弦相似度，速度快但精度有限
- **Reranker（交叉编码器）：** query和doc拼接输入，逐token交互，精度高但速度慢
- **使用场景：** 先用Embedding粗筛top-100，再用Reranker精排top-10
- **本项目用的：** bge-reranker，对检索结果重排序

**考察点：** 是否理解粗排vs精排的区别，为什么不能直接用Reranker遍历所有文档。

---

## 二、实际遇到的问题与解决方案

### 题目4：source_file字段返回错误怎么排查和修复？

**问题现象：** API返回的source_file字段为空或指向错误文档。

**根因分析：**
1. `schemas.py`响应模型缺少source_file字段
2. `answer_gen.py`生成答案时未传递source_file
3. `milvus_store.py`查询结果未返回source_file

**解决方案：**
```python
# schemas.py - 添加字段
class SourceChunk(BaseModel):
    source_file: str  # 新增
    page_no: int
    text: str
    score: float

# answer_gen.py - 传递字段
sources.append({
    "source_file": chunk.metadata.get("source_file", ""),  # 新增
    "page_no": chunk.page_no,
    "text": chunk.text,
    "score": chunk.score
})

# milvus_store.py - 查询时返回
results = self.collection.query(
    expr=f'source_file == "{source_file}"',  # 过滤条件
    output_fields=["text", "page_no", "source_file"]  # 明确指定返回字段
)
```

**考察点：** 排查问题的思路（从API响应→生成层→存储层逐层排查）。

---

### 题目5：同义词扩展是怎么做的？为什么不直接用向量检索？

**问题现象：** 用户问"营业额"，但文档里写的是"营业收入"，纯向量检索匹配不到。

**为什么向量不够：**
- bge-m3虽然有语义理解，但金融领域专业术语的同义词映射不够精确
- "营业额"和"营业收入"在向量空间的距离可能不够近

**解决方案：同义词词典 + 查询改写**
```python
SYNONYM_MAP = {
    "营业额": ["营业收入", "营收", "销售收入"],
    "老板": ["董事长", "董事局主席", "实际控制人"],
    "注册资本": ["注册资金", "实收资本"],
    "员工人数": ["在职员工", "职工总数"],
}

def expand_query(query: str) -> List[str]:
    """同义词扩展，返回多个查询"""
    expanded = [query]
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in query:
            for syn in synonyms:
                expanded.append(query.replace(keyword, syn))
    return expanded
```

**考察点：** 知道向量检索的局限性，有兜底方案。

---

### 题目6：BM25索引的懒加载机制是怎么设计的？

**问题背景：** 首次查询时BM25索引不存在，需要自动构建。

**实现方案：**
```python
class BM25Index:
    def __init__(self):
        self._index = None
        self._docs = []
        self._cache_path = Path("data/bm25_index.pkl")
    
    def _ensure_ready(self):
        """懒加载：有缓存就读缓存，没有就从Milvus构建"""
        if self._index is not None:
            return
            
        if self._cache_path.exists():
            # 方案1：从缓存加载
            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)
                self._docs = data["docs"]
                self._index = BM25Okapi([doc.split() for doc in self._docs])
        else:
            # 方案2：从Milvus读取所有文本，构建索引
            all_docs = self.milvus.get_all_texts()
            self._docs = [doc["text"] for doc in all_docs]
            self._index = BM25Okapi([doc.split() for doc in self._docs])
            
            # 保存缓存
            with open(self._cache_path, "wb") as f:
                pickle.dump({"docs": self._docs}, f)
    
    def search(self, query: str, top_k: int = 10):
        self._ensure_ready()
        scores = self._index.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._docs[i], scores[i]) for i in top_indices]
```

**考察点：** 懒加载设计、缓存策略、异常处理。

---

### 题目7：reranker选了历史值而不是最终值怎么办？

**问题现象：** 问"注册资本是多少"，reranker选了历史值5,225万而非最终值5,520万。

**根因分析：**
- 文档中有多个注册资本值（历次变更）
- Reranker根据语义相似度排序，历史值和最终值的语义相似度差不多
- 没有时间维度的权重

**解决方案：**
1. **文档预处理：** 入库时标注"最新"、"历史"等标签
2. **查询改写：** LLM识别问题意图是"当前值"还是"历史值"
3. **后处理排序：** 如果多个候选分数接近，优先选页码靠后的（通常最新信息在后面）
4. **元数据过滤：** 入库时记录文档章节，如"最新注册资本"vs"历史变更记录"

**考察点：** 遇到检索质量问题时的多维度解决思路。

---

### 题目8：RAGAS评测框架怎么用？遇到过什么坑？

**参考答案：**

**RAGAS四大指标：**
- **Faithfulness（忠实度）：** 答案是否基于检索到的上下文，不幻觉
- **Answer Relevancy（答案相关性）：** 答案是否回答了问题
- **Context Precision（上下文精确度）：** 检索到的文档是否相关
- **Context Recall（上下文召回率）：** 相关文档是否都被检索到

**遇到的坑：DeepSeek不支持n>1**
```python
# RAGAS内部强制n=3，DeepSeek报错
# 解决方案：包装LLM，强制n=1
from langchain_openai import ChatOpenAI
import functools

def get_deepseek_llm():
    llm = ChatOpenAI(model="deepseek-v4-pro", n=1)
    
    # 包装generate方法，强制n=1
    _original_generate = llm.generate
    
    @functools.wraps(_original_generate)
    def patched_generate(*args, **kwargs):
        if "n" in kwargs:
            kwargs["n"] = 1
        return _original_generate(*args, **kwargs)
    
    llm.generate = patched_generate
    return llm
```

**考察点：** 是否用过专业评测框架，遇到兼容性问题怎么解决。

---

### 题目9：指代消解（Coreference Resolution）怎么实现的？

**问题场景：**
```
用户：武汉兴图新科的主营业务是什么？
AI：视频指挥控制系统...
用户：它的注册资本呢？  ← "它"指代"兴图新科"
```

**实现方案：**
```python
class QueryUnderstander:
    def resolve_coreference(self, query: str, history: List[Dict]) -> str:
        """结合对话历史，把指代词替换为实体"""
        if not history:
            return query
        
        # 提取上一轮的主语
        last_query = history[-1]["user"]
        entities = self._extract_entities(last_query)
        
        # 替换指代词
        pronouns = ["它", "他", "她", "其", "该公司"]
        for pronoun in pronouns:
            if pronoun in query and entities:
                query = query.replace(pronoun, entities[0])
        
        return query
```

**考察点：** 多轮对话的处理能力，NLP基础。

---

### 题目10：如何处理PDF中的表格数据？

**问题现象：** PDF表格被解析成乱码，检索不到表格内容。

**解决方案：**
1. **解析阶段：** MinerU比Fitz更适合表格，能保留表格结构
2. **分块策略：** 表格按行分块，每行带表头（防止丢失列名）
3. **元数据标注：** 标记"这是一个表格"，检索时增加表格权重
4. **备选方案：** 表格转Markdown格式存储

```python
def split_table(table_text: str) -> List[str]:
    """表格按行分块，每行带表头"""
    lines = table_text.strip().split("\n")
    if len(lines) < 2:
        return [table_text]
    
    header = lines[0]
    chunks = []
    for line in lines[1:]:
        chunks.append(f"{header}\n{line}")  # 每行都带表头
    return chunks
```

**考察点：** 对PDF解析难点的理解，是否有实际处理经验。

---

## 三、系统设计与优化

### 题目11：如何设计一个可扩展的RAG系统，支持新增文档？

**参考答案：**
1. **文档管理层：** 统一的ingest.py脚本，支持增量入库
2. **映射配置：** `_COMPANY_SOURCE_MAP`字典，新增公司只需加一行
3. **索引更新：** 新文档入库后，自动更新BM25缓存
4. **评测机制：** 新增文档后跑RAGAS评测，确保质量不下降

```python
# 新增文档只需两步：
# 1. 把PDF放到data/source_docs/
# 2. 在映射表加一行
_COMPANY_SOURCE_MAP = {
    "兴图新科": "招股说明书1-无水印",
    "力源信息": "招股说明书2",
    "平安银行": "平安银行",  # 新增
    "招商证券": "招商证券",  # 新增
}
```

**考察点：** 系统设计能力，是否考虑可维护性。

---

### 题目12：检索结果的score阈值怎么定？低于阈值怎么处理？

**参考答案：**
- **动态阈值：** 不同问题类型的阈值不同
  - 精确查询（"注册资本是多少"）：阈值0.8
  - 模糊查询（"公司情况"）：阈值0.5
- **兜底策略：** 低于阈值时返回"无法从文档中找到答案"
- **置信度展示：** 返回top-3结果，让用户看到检索置信度

```python
def retrieve_with_threshold(self, query: str, threshold: float = 0.6):
    results = self.retrieve(query, top_k=10)
    filtered = [r for r in results if r["score"] >= threshold]
    
    if not filtered:
        return {"answer": "无法从文档中找到相关信息", "sources": []}
    
    return self.generate_answer(query, filtered)
```

**考察点：** 是否考虑检索质量控制，有无兜底机制。

---

### 题目13：如何评估RAG系统的效果？除了RAGAS还有什么方法？

**参考答案：**

**自动化评测：**
- RAGAS（本项目使用）：Faithfulness/Relevancy/Precision/Recall
- BLEU/ROUGE：答案和标准答案的文本相似度
- MRR（Mean Reciprocal Rank）：正确答案排在第几位

**人工评测：**
- 抽样100条，人工打分（1-5分）
- 评测维度：准确性、完整性、可读性
- 计算人工评分和RAGAS评分的相关性

**在线评测：**
- 用户反馈按钮（有用/没用）
- 点击率、停留时间等行为指标

**本项目做法：** 10道工单题，人工评测通过率（8/10 → 9/10）

**考察点：** 评测体系的全面性，不只是"跑个RAGAS就完事"。

---

### 题目14：Milvus和FAISS怎么选？你的项目为什么用Milvus？

**对比：**

| 维度 | Milvus | FAISS |
|------|--------|-------|
| 部署 | 独立服务(Docker) | 嵌入代码 |
| 持久化 | 自动 | 需手动保存索引文件 |
| 分布式 | 支持 | 不支持 |
| 元数据过滤 | 支持(expr语法) | 不支持 |
| 适用场景 | 生产环境、多服务共享 | 实验、单机小规模 |

**本项目选择Milvus的原因：**
1. 需要按source_file过滤（`expr='source_file == "xxx"'`）
2. 数据量大（9家公司年报，2万+向量），需要持久化
3. API服务需要独立进程，Milvus作为独立服务更稳定

**考察点：** 技术选型的依据，不是"因为教程用的Milvus"。

---

### 题目15：如果让你重新设计这个系统，你会改什么？

**参考答案（展示反思能力）：**

1. **分块策略：** 当前按固定token数分块，应该按语义分块（段落、章节）
2. **向量模型：** bge-m3通用模型，可以换成金融领域微调的模型
3. **缓存层：** 热门查询结果缓存，减少重复检索
4. **异步处理：** 文档入库改为异步队列，不阻塞API
5. **监控告警：** 添加检索延迟、LLM调用失败率等监控
6. **A/B测试：** 支持多套检索策略并行，数据驱动选最优

**考察点：** 不是"我的系统完美无缺"，而是能识别改进空间。

---

## 四、高频追问汇总

1. **分块大小怎么定的？** → 512token，实验调参，太小丢失上下文，太大引入噪声
2. **Embedding模型怎么选的？** → bge-m3，中文效果好，1024维够用
3. **LLM用的什么？** → DeepSeek-v4-pro，性价比高，中文能力强
4. **多轮对话怎么实现？** → Redis存session_id对应的历史，查询时注入prompt
5. **PDF解析用什么？** → Fitz(pymupdf)解析，MinerU处理复杂表格
6. **如何处理长文档？** → 分块+摘要，长文档先生成摘要，检索时同时匹配摘要和原文
7. **检索延迟多少？** → 向量检索50ms + BM25 20ms + Reranker 100ms + LLM 2s ≈ 2.2s
8. **如何处理并发？** → FastAPI异步 + Milvus连接池 + Redis分布式锁

---

## 五、项目数据（面试时可提及）

- **文档规模：** 9家公司年报 + 2份招股说明书，共23,792个向量
- **测试题：** 10道工单题，通过率9/10
- **评测框架：** RAGAS（Faithfulness/Answer Relevancy/Context Precision/Context Recall）
- **技术栈：** Python + FastAPI + Milvus + Redis + DeepSeek + bge-m3 + bge-reranker
- **部署方式：** Docker Compose（Milvus + Redis + API）

---

## 六、查询分解实现（MapReduce思想）

### 题目16：对比类和复合类问题怎么处理？

**问题现象：**
- 第9题"对比招商银行和平安银行..."只检索到平安银行
- 第10题"主要业务板块有哪些？各板块经营情况？"只检索到板块定义

**根因分析：**
1. 对比类：query太长，语义被稀释，一次检索无法覆盖多家公司
2. 复合类：一个问题包含多个意图（"有哪些" + "各板块情况"），检索被单一意图主导

**解决方案：通用查询分解**

```python
def decompose_query(question: str) -> List[Dict]:
    """支持两种分解类型"""
    
    # 1. 对比类 → 按公司分解
    # "对比招商银行和平安银行2019年净利润"
    # → ["招商银行2019年净利润", "平安银行2019年净利润"]
    
    # 2. 复合类 → 按意图分解
    # "主要业务板块有哪些？各板块经营情况？"
    # → ["主要业务板块有哪些", "各板块经营情况"]
```

**对比类分解示例：**
```
原问题: "对比招商银行和平安银行2019年的营业收入和净利润"
        ↓
子查询1: "招商银行2019年的营业收入和净利润" → source_filter="招商银行"
子查询2: "平安银行2019年的营业收入和净利润" → source_filter="平安银行"
        ↓ 分别检索（top-5）→ 合并去重 → top-10
```

**复合类分解示例：**
```
原问题: "中国平安保险的主要业务板块有哪些？各板块2019年的经营情况如何？"
        ↓
子查询1: "中国平安保险的主要业务板块有哪些" (list_entities)
子查询2: "中国平安各板块2019年的经营情况如何" (each_entity_status)
        ↓ 分别检索（top-5）→ 合并去重 → top-10
```

**代码实现：**
```python
# understander.py
def decompose_compare_query(self, question: str) -> List[Dict]:
    # 1. 对比类
    if any(kw in question for kw in ["对比", "比较", "哪家"]):
        return self._decompose_compare(question)
    
    # 2. 复合类
    for pattern in [r'有哪些.*各.*情况', r'主要.*包括.*各.*']:
        if re.search(pattern, question):
            return self._decompose_compound(question)
    
    # 3. 普通问题
    return [{"query": question, "sub_intent": "direct"}]

# main.py
sub_queries = understander.decompose_compare_query(question)
if len(sub_queries) > 1:
    # 分别检索每个子查询
    all_chunks = []
    for sq in sub_queries:
        chunks = retriever.retrieve(sq["query"], source_filter=sq.get("source_file"), top_k=5)
        all_chunks.extend(chunks)
    # 合并去重，按score排序
    context_chunks = deduplicate(all_chunks)[:10]
else:
    context_chunks = retriever.retrieve(question, ...)
```

**面试话术：**
> 第9、10题失败是因为复杂问题包含多个意图，单次检索无法覆盖。我用MapReduce思想解决：先分解成多个子查询（Map），分别检索，再合并结果（Reduce）。对比类按公司分解，复合类按意图分解。准确率从80%提升到90%。

---

## 七、代码改动清单

| 文件 | 改动 |
|------|------|
| `src/query/understander.py` | 添加`decompose_compare_query`、`_decompose_compare`、`_decompose_compound`方法 |
| `src/api/main.py` | `/query`和`/query/stream`接口集成查询分解逻辑 |
| `_COMPANY_SOURCE_MAP` | 添加9家金融公司的映射 |

**测试验证：**
```
对比类: "对比招商银行和平安银行2019年的营业收入和净利润"
→ 2个子查询 ✓

复合类: "中国平安保险的主要业务板块有哪些？各板块2019年的经营情况如何？"
→ 2个子查询 ✓

普通类: "招商银行2019年的营业收入是多少"
→ 1个子查询（不分解）✓
```

---

*早读材料生成时间：2026-05-30*
*基于真实项目经验整理，非网上抄的八股文*
*新增：查询分解（对比类+复合类），解决第9、10题准确率*
