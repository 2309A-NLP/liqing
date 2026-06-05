# RAG 多角色扮演系统

一个基于 FastAPI + Milvus + Redis + DeepSeek 的中文多角色 RAG 项目。

## 快速开始

```bash
pip install -r requirements.txt
# 从项目根目录运行：

```

然后打开前端页面 `研发/frontend/index.html`。

## 文档索引

- `DEPLOYMENT.md`：Windows / Docker 部署、启动与排错
- `.env.example`：环境变量示例与说明
- `requirements.txt`：基础依赖清单
- `requirements-lock.txt`：锁定版本的依赖清单
- `RAG性能优化说明.md`：最近几轮 RAG 性能优化思路与结果
- `模型作用说明.md`：项目中各模型/组件的职责说明
- `目录索引.md`：部署文档目录总览
- `notes-streaming.txt`：流式回答实现备忘

## 核心特性

- 向量检索 + BM25 检索 + rerank
- 短期会话记忆 + 长期记忆
- 多角色提示词
- 前端性能面板与检索结果可视化
- 基准测试脚本 `RAG性能基准测试.py`
