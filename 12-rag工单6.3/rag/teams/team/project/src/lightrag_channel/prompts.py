"""
LightRAG 通道 — 金融文档实体类型定制
工单12：LightRAG 优化任务

针对招股说明书等金融文档定制实体类型和抽取提示。
"""

# 金融文档专用实体类型 guidance
FINANCIAL_ENTITY_TYPES_GUIDANCE = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: 个人（高管、股东、法定代表人等）
- Organization: 公司、机构、政府部门
- Subsidiary: 子公司、控股公司、关联方
- FinancialMetric: 财务指标（净利润、毛利率、总资产、营业收入、资产负债率等）
- IPOProject: 募投项目、募集资金用途
- BusinessSegment: 业务板块、主营业务分类
- Product: 产品、服务、技术
- RiskFactor: 风险因素
- RegulatoryBody: 监管机构（证监会、交易所等）
- Shareholder: 股东、实际控制人
- LegalDocument: 法律文件、合同、协议
- Location: 地理位置（注册地、经营地）
- Event: 事件（上市、融资、诉讼等）
- Data: 数值数据（发行股数、持股比例、金额等）
- Concept: 概念、政策、行业术语"""

# 金融文档实体抽取的 few-shot 示例
FINANCIAL_ENTITY_EXTRACTION_EXAMPLES = """---Entity Types---
- Person: 个人
- Organization: 公司、机构
- Subsidiary: 子公司、关联方
- FinancialMetric: 财务指标
- IPOProject: 募投项目
- BusinessSegment: 业务板块
- Product: 产品、服务
- RiskFactor: 风险因素
- RegulatoryBody: 监管机构
- Shareholder: 股东
- Data: 数值数据
- Concept: 概念、术语

---Input Text---
```
武汉力源信息技术股份有限公司本次公开发行人民币普通股（A股）1,700万股，
占发行后总股本的25.37%。募集资金拟投资于以下项目：
1. 电子元器件分销网络扩建项目，投资总额12,000万元；
2. 技术研发中心建设项目，投资总额3,000万元。
公司2009年实现营业收入82,345.67万元，净利润3,456.78万元。
```

---Output---
entity<|#|>武汉力源信息技术股份有限公司<|#|>Organization<|#|>武汉力源信息技术股份有限公司是一家电子元器件分销企业，本次公开发行A股股票。
entity<|#|>人民币普通股<|#|>Concept<|#|>人民币普通股（A股）是公司本次公开发行的股票类型。
entity<|#|>电子元器件分销网络扩建项目<|#|>IPOProject<|#|>电子元器件分销网络扩建项目是公司募集资金拟投资项目之一，投资总额12,000万元。
entity<|#|>技术研发中心建设项目<|#|>IPOProject<|#|>技术研发中心建设项目是公司募集资金拟投资项目之一，投资总额3,000万元。
entity<|#|>营业收入<|#|>FinancialMetric<|#|>营业收入是衡量公司经营规模的核心指标，公司2009年营业收入为82,345.67万元。
entity<|#|>净利润<|#|>FinancialMetric<|#|>净利润是衡量公司盈利能力的核心指标，公司2009年净利润为3,456.78万元。
entity<|#|>1,700万股<|#|>Data<|#|>公司本次公开发行1,700万股A股，占发行后总股本的25.37%。
entity<|#|>25.37%<|#|>Data<|#|>本次公开发行股数占发行后总股本的比例为25.37%。
relation<|#|>武汉力源信息技术股份有限公司<|#|>电子元器件分销网络扩建项目<|#|>募集资金投资<|#|>公司募集资金拟投资于电子元器件分销网络扩建项目，投资总额12,000万元。
relation<|#|>武汉力源信息技术股份有限公司<|#|>技术研发中心建设项目<|#|>募集资金投资<|#|>公司募集资金拟投资于技术研发中心建设项目，投资总额3,000万元。
relation<|#|>武汉力源信息技术股份有限公司<|#|>营业收入<|#|>财务指标<|#|>公司2009年营业收入为82,345.67万元。
relation<|#|>武汉力源信息技术股份有限公司<|#|>净利润<|#|>财务指标<|#|>公司2009年净利润为3,456.78万元。
relation<|#|>武汉力源信息技术股份有限公司<|#|>人民币普通股<|#|>发行股票<|#|>公司本次公开发行人民币普通股（A股）1,700万股。
<|COMPLETE|>

"""
