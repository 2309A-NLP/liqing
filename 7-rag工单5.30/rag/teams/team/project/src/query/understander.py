"""
Query 理解模块 — LLM 驱动的问题改写与分解
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

功能：
  1. 指代消解：结合对话历史，把"它的收入呢"改写成完整问题
  2. 问题分解：复杂问题拆成多个子问题分别检索
  3. 意图提取：识别问题核心实体和意图，用于日志分析
  4. 查询分解：对比类问题拆成多个独立查询（MapReduce思想）
"""

import json
import time
import logging
import re
from typing import List, Dict, Any
import httpx
from src.config import config

logger = logging.getLogger("rag")


class QueryUnderstander:
    """LLM 驱动的 Query 理解器

    输入：原始问题 + 对话历史
    输出：改写后的问题 + 子问题列表 + 意图标签 + source_file 过滤
    """

    # 公司名关键词 → source_file 映射
    # 新增文档时在这里加一条即可
    _COMPANY_SOURCE_MAP = {
        "兴图新科": "招股说明书1-无水印",
        "武汉兴图新科": "招股说明书1-无水印",
        "力源信息": "招股说明书2",
        "武汉力源": "招股说明书2",
        "力源科技": "招股说明书2",
        # 9家金融公司
        "平安银行": "平安银行",
        "招商证券": "招商证券",
        "中信证券": "中信证券",
        "中国人寿": "中国人寿保险",
        "中国人寿保险": "中国人寿保险",
        "中国平安": "中国平安保险",
        "中国平安保险": "中国平安保险",
        "邮政储蓄": "中国邮政储蓄银行",
        "中国邮政储蓄银行": "中国邮政储蓄银行",
        "邮储银行": "中国邮政储蓄银行",
        "国泰君安": "国泰君安证券",
        "国泰君安证券": "国泰君安证券",
        "太平洋保险": "太平洋保险",
        "中国太保": "太平洋保险",
        "招商银行": "招商银行",
    }

    # 对比类关键词
    _COMPARE_KEYWORDS = ["对比", "比较", "哪家", "哪个更", "vs", "VS", "相比", "对照"]

    # 复合问题关键词（包含多个意图）
    _COMPOUND_PATTERNS = [
        (r'有哪些.*各.*情况', 'list_and_detail'),
        (r'主要.*哪些.*各.*', 'list_and_detail'),
        (r'主要.*包括.*各.*', 'list_and_detail'),
        (r'包括.*各.*经营', 'list_and_detail'),
        (r'哪些.*各.*数据', 'list_and_detail'),
        (r'分别.*各.*', 'list_and_detail'),
    ]

    # 代词列表
    _PRONOUNS = ["它", "他", "她", "其", "该公司", "这家", "该企业"]

    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = config.DEEPSEEK_BASE_URL
        self.model = config.DEEPSEEK_MODEL

    def extract_source_filter(
        self,
        question: str,
        history: List[Dict[str, Any]] | None = None,
    ) -> str | None:
        """从问题（和历史）中提取目标文档的 source_file

        逻辑：
        1. 问题含对比/比较多公司 → 返回 None（搜全部）
        2. 当前问题提到一家公司名 → 返回对应 source_file
        3. 当前问题有代词但历史里有公司名 → 返回历史公司对应的 source_file
        4. 都没有 → 返回 None（不做过滤，搜全部）
        """
        # 对比类问题：提到多家公司，不做过滤
        is_compare = any(kw in question for kw in self._COMPARE_KEYWORDS)

        # 统计问题中提到的公司数量
        matched_companies = []
        for keyword, source_file in self._COMPANY_SOURCE_MAP.items():
            if keyword in question:
                if source_file not in matched_companies:
                    matched_companies.append(source_file)

        # 多家公司或对比类问题 → 搜全部
        if len(matched_companies) > 1 or (is_compare and matched_companies):
            return None

        # 单家公司 → 过滤
        if matched_companies:
            return matched_companies[0]

        # 当前问题没有公司名，检查是否有代词
        has_pronoun = any(p in question for p in self._PRONOUNS)
        if has_pronoun and history:
            # 从历史中找最近提到的公司名
            for h in reversed(history[-6:]):
                text = h.get("content", "")
                for keyword, source_file in self._COMPANY_SOURCE_MAP.items():
                    if keyword in text:
                        return source_file

        return None

    def extract_retrieval_query(self, question: str) -> str:
        """提取检索专用query：去掉公司名前缀，保留核心意图

        解决问题：长query中的公司全称把向量检索带偏，核心意图被稀释。
        例："根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"
        → "电子信息行业的上游涉及哪些企业？"

        返回核心意图query，用于向量检索。原始query仍保留用于BM25和答案生成。
        """
        q = question

        # 去掉"根据XX招股意向书/招股说明书，"前缀
        q = re.sub(r'^根据[^，,。]+(?:招股意向书|招股说明书|募集说明书)[，,]?\s*', '', q)

        # 去掉"与XX公司/武汉XX"开头的介词短语（保留后面的核心动词）
        q = re.sub(r'^与(?:武汉|上海|北京|深圳|广州|杭州|成都|南京)?\S{2,}(?:股份)?(?:有限公司|集团|公司)[的]?', '', q)

        # 去掉"报告期内，XX公司"前缀
        q = re.sub(r'^报告期内[，,]?(?:武汉|上海|北京|深圳)?\S{2,}(?:股份)?(?:有限公司|公司)[的]?', '', q)

        # 去掉"武汉XX公司的"前缀
        q = re.sub(r'^(?:武汉|上海|北京|深圳|广州)\S{2,}(?:股份)?(?:有限公司|公司)[的]?', '', q)

        # 清理开头的标点和空白
        q = q.lstrip('，,。、 ')

        # 如果清理后太短或为空，回退到原query
        if len(q) < 4:
            return question

        return q

    def decompose_compare_query(self, question: str) -> List[Dict[str, Any]]:
        """查询分解（MapReduce思想）

        支持两种类型：
        1. 对比类问题：拆成多个公司独立查询
        2. 复合类问题：拆成多个意图独立查询

        Args:
            question: 原始问题

        Returns:
            List[Dict]: 子查询列表，每个元素包含:
                - query: 子查询文本
                - source_file: 目标文档过滤（可选）
                - company: 公司名
                - sub_intent: 子意图标签

        示例:
            对比类: "对比招商银行和平安银行2019年的营业收入和净利润"
            → [
                {"query": "招商银行2019年的营业收入和净利润", "source_file": "招商银行", "sub_intent": "compare"},
                {"query": "平安银行2019年的营业收入和净利润", "source_file": "平安银行", "sub_intent": "compare"},
            ]

            复合类: "中国平安保险的主要业务板块有哪些？各板块2019年的经营情况如何？"
            → [
                {"query": "中国平安保险主要业务板块有哪些", "sub_intent": "list_entities"},
                {"query": "中国平安保险各业务板块2019年经营情况", "sub_intent": "each_entity_status"},
            ]
        """
        # 1. 检测对比类问题
        is_compare = any(kw in question for kw in self._COMPARE_KEYWORDS)
        if is_compare:
            return self._decompose_compare(question)

        # 2. 检测复合类问题（包含多个意图）
        for pattern, intent in self._COMPOUND_PATTERNS:
            if re.search(pattern, question):
                return self._decompose_compound(question, intent)

        # 3. 普通问题，直接返回
        return [{"query": question, "source_file": None, "company": None, "sub_intent": "direct"}]

    def _decompose_compare(self, question: str) -> List[Dict[str, Any]]:
        """对比类问题分解"""

        # 提取问题中提到的公司
        matched_companies = []
        for keyword, source_file in self._COMPANY_SOURCE_MAP.items():
            if keyword in question:
                if source_file not in [c["source_file"] for c in matched_companies]:
                    matched_companies.append({
                        "keyword": keyword,
                        "source_file": source_file,
                    })

        # 只有一家公司或没有公司，不需要分解
        if len(matched_companies) < 2:
            return [{"query": question, "source_file": None, "company": None, "sub_intent": "direct"}]

        # 提取核心问题（去掉公司名和对比词）
        core_question = question
        for company_info in matched_companies:
            core_question = core_question.replace(company_info["keyword"], "")
        for kw in self._COMPARE_KEYWORDS:
            core_question = core_question.replace(kw, "")
        core_question = core_question.strip("，,。、的和与 ")

        # 如果核心问题太短，保留原问题的后半部分
        if len(core_question) < 4:
            match = re.search(r'[的]([^，,。？?]+[？?]?)$', question)
            if match:
                core_question = match.group(1).strip()
            else:
                core_question = "相关信息"

        # 生成子查询
        sub_queries = []
        for company_info in matched_companies:
            sub_q = f"{company_info['keyword']}{core_question}"
            sub_queries.append({
                "query": sub_q,
                "source_file": company_info["source_file"],
                "company": company_info["keyword"],
                "sub_intent": "compare",
            })

        logger.info(f"[查询分解] 对比类问题: {question}")
        logger.info(f"[查询分解] 分解为 {len(sub_queries)} 个子查询:")
        for i, sq in enumerate(sub_queries):
            logger.info(f"  [{i+1}] {sq['query']} → {sq['source_file']}")

        return sub_queries

    def _decompose_compound(self, question: str, intent: str) -> List[Dict[str, Any]]:
        """复合类问题分解

        例: "中国平安保险的主要业务板块有哪些？各板块2019年的经营情况如何？"
        分解为:
        1. "中国平安保险主要业务板块有哪些" (list_entities)
        2. "中国平安保险各业务板块2019年经营情况" (each_entity_status)
        """

        # 提取公司名
        company = None
        for keyword, source_file in self._COMPANY_SOURCE_MAP.items():
            if keyword in question:
                company = keyword
                break

        # 构造子查询
        sub_queries = []

        # 子问题1：有哪些板块/类型
        # 从原问题中提取"有哪些"或"包括哪些"部分
        list_match = re.search(r'([^，。？]*(?:有哪些|包括哪些)[^，。？]*)', question)
        if list_match:
            sub_q1 = list_match.group(1).strip()
            if company and company not in sub_q1:
                sub_q1 = f"{company}{sub_q1}"
            sub_queries.append({
                "query": sub_q1,
                "source_file": self._COMPANY_SOURCE_MAP.get(company),
                "company": company,
                "sub_intent": "list_entities",
            })

        # 子问题2：各板块经营情况
        detail_match = re.search(r'(各[^，。？]*情况[^，。？]*)', question)
        if detail_match:
            sub_q2 = detail_match.group(1).strip()
            if company and company not in sub_q2:
                sub_q2 = f"{company}{sub_q2}"
            # 提取年份
            year_match = re.search(r'(\d{4})年', question)
            if year_match and year_match.group(1) not in sub_q2:
                sub_q2 = f"{year_match.group(1)}年{sub_q2}"
            sub_queries.append({
                "query": sub_q2,
                "source_file": self._COMPANY_SOURCE_MAP.get(company),
                "company": company,
                "sub_intent": "each_entity_status",
            })

        # 如果没有匹配到，返回原问题
        if not sub_queries:
            return [{"query": question, "source_file": None, "company": None, "sub_intent": "direct"}]

        logger.info(f"[查询分解] 复合类问题: {question}")
        logger.info(f"[查询分解] 分解为 {len(sub_queries)} 个子查询:")
        for i, sq in enumerate(sub_queries):
            logger.info(f"  [{i+1}] {sq['query']} (intent={sq['sub_intent']})")

        return sub_queries

    def understand(
        self,
        question: str,
        history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """理解用户问题

        Returns:
            {
                "original": str,          # 原始问题
                "rewritten": str,         # 改写后的完整问题
                "sub_questions": [str],   # 分解的子问题（用于多路检索）
                "intent": str,            # 意图标签
            }
        """
        t0 = time.perf_counter()

        # 无历史 → 跳过改写，直接返回
        if not history:
            result = {
                "original": question,
                "rewritten": question,
                "sub_questions": [question],
                "intent": "direct",
            }
            logger.info(f"[Query理解] 无历史，跳过改写 | intent=direct")
            return result

        # 有历史 → 调用 LLM 改写
        try:
            result = self._call_llm_rewrite(question, history)
        except Exception as e:
            logger.warning(f"[Query理解] LLM 改写失败，回退到原始问题: {e}")
            result = {
                "original": question,
                "rewritten": question,
                "sub_questions": [question],
                "intent": "fallback",
            }

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Query理解] {elapsed:.0f}ms | intent={result['intent']} | "
            f"rewritten={result['rewritten'][:60]}"
        )
        return result

    def _call_llm_rewrite(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """调用 LLM 进行问题改写"""
        # 构建对话历史文本
        history_text = ""
        for h in history[-6:]:
            role = "用户" if h.get("role") == "user" else "助手"
            content = h.get("content", "")[:200]
            history_text += f"{role}: {content}\n"

        prompt = f"""请分析以下用户问题，结合对话历史进行改写和分解。

对话历史：
{history_text}

当前问题：{question}

请返回 JSON 格式：
{{
    "rewritten": "改写后的完整问题（解决指代消解，如'它的收入'→'XX公司的收入'）",
    "sub_questions": ["子问题1", "子问题2"],
    "intent": "direct/follow_up/compare/complex"
}}

注意：
1. 如果问题中有"它"、"其"等代词，请根据历史替换为具体实体
2. 如果问题涉及对比（如"A和B哪个大"），分解为两个子问题
3. 如果问题很简单，sub_questions 只包含改写后的问题
4. 只返回 JSON，不要其他文字"""

        messages = [{"role": "user", "content": prompt}]

        # 调用 DeepSeek API
        response_text = self._call_api(messages)

        # 解析 JSON
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "original": question,
                    "rewritten": data.get("rewritten", question),
                    "sub_questions": data.get("sub_questions", [question]),
                    "intent": data.get("intent", "direct"),
                }
        except json.JSONDecodeError:
            pass

        # 解析失败，返回原始问题
        return {
            "original": question,
            "rewritten": question,
            "sub_questions": [question],
            "intent": "parse_error",
        }

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用 DeepSeek API"""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
