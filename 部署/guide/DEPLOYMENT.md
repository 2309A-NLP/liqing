# 部署说明

本文档用于指导在 Windows 本地或 Docker 环境中部署本项目，并提供常见问题排查方法。

## 一、Windows 本地启动

### 1. 安装依赖

```bash
pip install -r 部署/requirements/requirements-lock.txt
```

### 2. 启动后端

从项目根目录（`rag-multi-role/`）运行：

```bash
# 推荐：Linux / macOS
PYTHONPATH=研发 python -m online.app

# Windows CMD
set PYTHONPATH=研发 && python -m online.app

# Windows PowerShell
$env:PYTHONPATH="研发"; python -m online.app
```

默认监听地址为 `http://127.0.0.1:8002`。

### 3. 打开前端

直接打开 `研发/frontend/index.html` 即可。

---

## 二、Docker 启动

### 1. 构建镜像

```bash
docker build -t rag-multi-role .
```

### 2. 启动容器

```bash
docker run --rm -p 8002:8002 --env-file .env rag-multi-role
```

---

## 三、环境变量

环境变量示例见 `部署/config/.env.example`。

### 关键项

- `DEEPSEEK_API_KEY`：DeepSeek 接口密钥
- `DEEPSEEK_BASE_URL`：OpenAI 兼容接口地址
- `DEEPSEEK_MODEL`：生成模型名
- `REDIS_URL`：Redis 地址
- `REDIS_PASSWORD`：Redis 密码
- `MILVUS_HOST`：Milvus 地址
- `MILVUS_PORT`：Milvus 端口
- `MILVUS_DATABASE`：Milvus 数据库名
- `MILVUS_COLLECTION`：知识库集合名
- `MILVUS_MEMORY_COLLECTION`：长期记忆集合名
- `EMBED_MODEL_PATH`：向量模型路径
- `RERANK_MODEL_PATH`：重排模型路径

---

## 四、常见排查

### 1. 相对导入报错

请从项目根目录运行，并设置 `PYTHONPATH`（见上方启动命令）。不要直接运行单个 `.py` 文件。

### 2. 依赖安装失败

确认使用的是 `requirements-lock.txt`，并检查 Python 版本是否兼容。

### 3. Redis / Milvus 连接失败

- 检查服务是否已启动
- 检查地址、端口、用户名、密码是否正确

### 4. 模型加载失败

- 检查 `EMBED_MODEL_PATH` 和 `RERANK_MODEL_PATH`
- 检查模型目录是否真实存在
- 检查磁盘权限与路径格式（Windows 下注意反斜杠）

### 5. DeepSeek 调用失败

- 检查 `DEEPSEEK_API_KEY`
- 检查 `DEEPSEEK_BASE_URL`
- 检查网络是否可访问外部 API
