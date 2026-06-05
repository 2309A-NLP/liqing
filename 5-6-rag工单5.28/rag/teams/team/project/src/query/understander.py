"""
Query 理解模块 — LLM 驱动的问题改写与分解
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

功能：
  1. 指代消解：结合对话历史，把"它的收入呢"改写成完整问题
  2. 问题分解：复杂问题拆成多个子问题分别检索
  3. 意图提取：识别问题核心实体和意图，用于日志分析
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
    }

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
        1. 当前问题提到公司名 → 返回对应 source_file
        2. 当前问题有代词但历史里有公司名 → 返回历史公司对应的 source_file
        3. 都没有 → 返回 None（不做过滤，搜全部）
        """
        # 先从当前问题提取
        for keyword, source_file in self._COMPANY_SOURCE_MAP.items():
            if keyword in question:
                return source_file

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
        import re
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

        # 有历史 → LLM 改写
        result = self._llm_understand(question, history)
        ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[Query理解] {ms:.0f}ms | intent={result['intent']} | "
            f"原始='{question[:50]}' → 改写='{result['rewritten'][:50]}'"
        )
        if result["sub_questions"] and len(result["sub_questions"]) > 1:
            for i, sq in enumerate(result["sub_questions"]):
                logger.info(f"  子问题#{i+1}: {sq}")

        return result

    # 需要消解的代词
    _PRONOUNS = {"它", "该公司", "其", "该企业", "这家", "这个公司", "这个企业", "此公司"}

    def _llm_understand(
        self,
        question: str,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """调用 LLM 进行 Query 理解"""

        # 构造历史摘要（最近 6 条 = 3 轮）
        history_text = ""
        for h in history[-4:]:  # 最近 4 条 = 2 轮（减少 token，加快响应）
            role = "用户" if h.get("role") == "user" else "助手"
            content = h.get("content", "")[:200]
            history_text += f"{role}: {content}\n"

        prompt = f"""你是一个查询改写引擎。你的核心任务是把用户问题改写成独立可检索的完整问句。

## 最重要的规则：指代消解
如果当前问题含有"它"、"该公司"、"其"、"该企业"、"这家"、"这个公司"等指代词，
你必须从对话历史中找到指代的具体实体，然后替换掉指代词。

例如：
- 历史中用户问过"武汉兴图新科电子股份有限公司的法定代表人是谁"
- 当前问题："它的注册资本是多少"
- 你必须输出："武汉兴图新科电子股份有限公司的注册资本是多少"

- 历史中用户问过"第四节的内容是什么"
- 当前问题："那它的第一条是什么"
- 你必须输出："招股说明书第四节的第一条是什么"

## 任务
1. 指代消解（必须做！把所有代词替换为具体名称）
2. 问题分解（如果含多个子问题则拆分）
3. 意图分类（fact_lookup / comparison / summary / explanation / follow_up）

## 对话历史
{history_text}

## 当前问题
{question}

## 输出格式（严格 JSON，不要输出其他内容）
{{"rewritten": "改写后的完整独立问句", "sub_questions": ["子问题1"], "intent": "意图标签"}}"""

        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self._call_api(messages)
            parsed = self._parse_json(raw)
            result = {
                "original": question,
                "rewritten": parsed.get("rewritten", question),
                "sub_questions": parsed.get("sub_questions", [question]),
                "intent": parsed.get("intent", "unknown"),
            }
        except Exception as e:
            logger.warning(f"[Query理解] LLM 解析失败，回退原始问题: {e}")
            result = {
                "original": question,
                "rewritten": question,
                "sub_questions": [question],
                "intent": "fallback",
            }

        # ── 兜底：检查是否还有代词没消解 ──
        if self._has_pronoun(result["rewritten"]):
            entity = self._extract_entity(history)
            if entity:
                logger.info(f"[Query理解] 检测到未消解代词，补刀 entity={entity[:30]}")
                result = self._retry_with_entity(question, result, entity)

        return result

    def _has_pronoun(self, text: str) -> bool:
        """检查文本中是否还有未消解的代词"""
        return any(p in text for p in self._PRONOUNS)

    def _extract_entity(self, history: List[Dict[str, Any]]) -> str:
        """从历史对话中提取主实体（公司名、项目名等）"""
        import re
        # 扫描历史用户消息，找最长的实体名
        for h in history[-4:]:
            if h.get("role") != "user":
                continue
            text = h.get("content", "")
            # 匹配公司名：XX有限公司 / XX股份有限公司
            m = re.search(r"[\u4e00-\u9fa5]{2,}(?:股份)?有限公司", text)
            if m:
                return m.group(0)
            # 匹配 XX 科技 / XX 集团
            m = re.search(r"[\u4e00-\u9fa5]{2,}(?:科技|集团|控股)", text)
            if m:
                return m.group(0)
        return ""

    def _retry_with_entity(
        self,
        question: str,
        first_result: Dict[str, Any],
        entity: str,
    ) -> Dict[str, Any]:
        """用明确实体名重新改写"""
        prompt = f"""## 任务
将以下问题中的代词替换为实体名称，输出一个完整的独立问句。

## 实体名称
{entity}

## 原始问题
{question}

## 替换规则（严格遵守！）
1. 只替换代词："它"→"{entity}"、"该公司"→"{entity}"、"其"→"{entity}"、"该企业"→"{entity}"
2. 问题的其余部分一字不改
3. 不要添加任何额外内容
4. 只输出替换后的问句，不要输出解释

## 示例
原始问题："它的注册资本是多少？"
输出："{entity}的注册资本是多少？"

原始问题："它的董事长是谁"
输出："{entity}的董事长是谁"

原始问题："它上市了吗"
输出："{entity}上市了吗"

原始问题："它在哪个领域已经成为重要供应商？"
输出："{entity}在哪个领域已经成为重要供应商？"

现在请替换："""

        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self._call_api(messages).strip().strip('"').strip("'")
            if raw and not self._has_pronoun(raw):
                first_result["rewritten"] = raw
                first_result["sub_questions"] = [raw]
                logger.info(f"[Query理解] 补刀成功 → '{raw[:60]}'")
            else:
                logger.warning(f"[Query理解] 补刀结果仍有代词，保留原结果")
        except Exception as e:
            logger.warning(f"[Query理解] 补刀失败: {e}")

        return first_result

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """解析 LLM 返回的 JSON（兼容 markdown 代码块）"""
        json_str = raw.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        return json.loads(json_str)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用 LLM API（轻量级，低温度，短超时）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # 极低温度，追求确定性
            "max_tokens": 256,   # 短输出，够用就行
        }

        with httpx.Client(timeout=10) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
