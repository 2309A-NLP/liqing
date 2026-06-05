"""
文档隔离检索 — 按公司名过滤查询结果
工单12：LightRAG 优化任务

解决两份招股书图谱合并后，跨文档实体混淆的问题。
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("lightrag_channel")

# ── 公司元数据 ──
COMPANY_META = {
    "力源信息": {
        "full_name": "武汉力源信息技术股份有限公司",
        "aliases": ["力源信息", "力源", "武汉力源"],
        "doc_id": "招股说明书2",
        "distinct_keywords": [
            "IC", "目录销售", "分销", "半导体", "赵马克", "Mark Zhao",
            "渠道销售", "电话及网络销售", "大客户销售", "国际贸易",
            "仓储物流", "电子商务平台", "普芯达", "佰力电子",
            "融冰投资", "听音投资", "联众聚源", "武汉博润", "上海博润",
            "力源贸易", "香港力源", "武汉经发",
        ],
    },
    "兴图新科": {
        "full_name": "武汉兴图新科电子股份有限公司",
        "aliases": ["兴图新科", "兴图", "武汉兴图"],
        "doc_id": "招股说明书1-无水印",
        "distinct_keywords": [
            "视频指挥", "视音频", "军用", "军方", "国防", "程家明",
            "指挥控制", "预警控制", "编码器", "解码器", "显控",
            "云联邦", "监狱", "油田", "视频会议", "多媒体调度",
            "网络化视频指挥系统", "综合管理服务器",
        ],
    },
}


def detect_company(question: str) -> Optional[str]:
    """检测问题中提到的公司名

    Returns:
        "力源信息" / "兴图新科" / None（未明确提到某家公司）
    """
    q = question.strip()

    for company, meta in COMPANY_META.items():
        for alias in meta["aliases"]:
            if alias in q:
                logger.info(f"[文档隔离] 检测到公司: {company}")
                return company

    # 通过特征关键词推断
    for company, meta in COMPANY_META.items():
        score = sum(1 for kw in meta["distinct_keywords"] if kw in q)
        if score >= 2:
            logger.info(f"[文档隔离] 通过关键词推断公司: {company} (匹配{score}个关键词)")
            return company

    logger.info(f"[文档隔离] 未检测到具体公司")
    return None


def filter_context_by_company(
    contexts: List[str],
    company: str,
) -> List[str]:
    """过滤上下文，只保留与目标公司相关的chunk

    Args:
        contexts: LightRAG返回的上下文列表
        company: 目标公司名

    Returns:
        过滤后的上下文列表
    """
    if not company or company not in COMPANY_META:
        return contexts

    meta = COMPANY_META[company]
    other_companies = [c for c in COMPANY_META if c != company]

    filtered = []
    for ctx in contexts:
        # 检查是否包含其他公司的特征关键词
        is_other = False
        for other in other_companies:
            other_meta = COMPANY_META[other]
            # 如果包含其他公司的全名或大量其他公司关键词，排除
            if other_meta["full_name"] in ctx:
                other_score = sum(1 for kw in other_meta["distinct_keywords"][:5] if kw in ctx)
                my_score = sum(1 for kw in meta["distinct_keywords"][:5] if kw in ctx)
                if other_score > my_score:
                    is_other = True
                    break

        if not is_other:
            filtered.append(ctx)

    if len(filtered) < len(contexts):
        logger.info(f"[文档隔离] 过滤: {len(contexts)} → {len(filtered)} 个上下文 (公司={company})")

    # 如果过滤后没有内容，返回原始上下文（防止空结果）
    return filtered if filtered else contexts


def rewrite_question_for_company(question: str, company: str) -> str:
    """在问题中显式加入公司全名，帮助LightRAG精准检索

    如果问题中已经包含公司名，不改写。
    """
    if not company or company not in COMPANY_META:
        return question

    meta = COMPANY_META[company]

    # 检查是否已经有公司全名
    if meta["full_name"] in question:
        return question

    # 检查是否有简称
    for alias in meta["aliases"]:
        if alias in question:
            # 曫换简称为全名，帮助图谱检索
            rewritten = question.replace(alias, meta["full_name"])
            logger.info(f"[文档隔离] 问题改写: {question[:50]}... → {rewritten[:50]}...")
            return rewritten

    # 没有公司名，在前面加上公司名前缀
    rewritten = f"关于{meta['full_name']}：{question}"
    logger.info(f"[文档隔离] 问题加前缀: {rewritten[:60]}...")
    return rewritten
