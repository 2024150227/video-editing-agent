# 基于多模态大模型的智能视频剪辑 Agent

基于多模态大模型（AgnesAPI）的智能视频剪辑系统。用户提供原始素材和剪辑需求，系统自动完成分镜脚本生成、素材智能匹配、视频自动剪辑与渲染的全流程。

## 系统架构

```
用户需求 ──► 导演 Agent ──► 分镜脚本
                    │
                    ▼
              素材 Agent ──► 素材匹配/生成
                    │
                    ▼
              剪辑 Agent ──► 视频合成
                    │
                    ▼
              渲染 Worker ──► 输出视频
```

| 组件 | 职责 |
|------|------|
| **导演 Agent** (`app/agents/director.py`) | 接收用户需求，调用 LLM 生成结构化分镜脚本 |
| **素材 Agent** (`app/agents/material.py`) | 根据分镜检索本地素材库，必要时调用 AI 生成补充素材 |
| **剪辑 Agent** (`app/agents/editor.py`) | 编排剪辑任务，协调渲染流程 |
| **渲染 Worker** (`app/workers/render_worker.py`) | Celery 异步任务，使用 MoviePy 完成视频合成与字幕叠加 |
| **向量检索** (`app/services/index_service.py`) | 使用 CLIP 模型对素材进行语义嵌入与 Qdrant 向量检索 |

## 技术栈

- **后端框架**: FastAPI (Python 3.11+)
- **多模态 LLM**: AgnesAPI (agnes-2.0-flash, 免费)
- **异步任务**: Celery + Redis
- **向量数据库**: Qdrant + CLIP 嵌入
- **视频处理**: MoviePy / FFmpeg
- **数据库**: PostgreSQL (SQLAlchemy async)
- **测试**: pytest + httpx (AsyncClient)

## 快速启动

### 前置条件

- Python 3.11+
- Docker & Docker Compose (用于启动 PostgreSQL / Redis / Qdrant)
- AgnesAPI Key (https://platform.agnes-ai.com/)

### 1. 启动依赖服务

```bash
docker-compose up -d
```

此命令将启动 PostgreSQL (5432)、Redis (6379) 和 Qdrant (6333)。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，至少填入 `AGNES_API_KEY`。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

推荐在虚拟环境中安装：

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 4. 数据库迁移

```bash
alembic upgrade head
```

### 5. 启动 API 服务

```bash
uvicorn app.main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看交互式 API 文档。

### 6. 启动 Celery Worker

打开新的终端窗口：

```bash
celery -A app.core.celery_app worker --loglevel=info --concurrency=1
```

## API 使用流程

### 1. 创建项目

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "旅行vlog", "material_source_dir": "/path/to/materials"}'
```

### 2. 生成分镜脚本

```bash
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/story \
  -H "Content-Type: application/json" \
  -d '{"prompt": "旅行素材30秒快剪", "style": "快节奏", "total_duration_sec": 30}'
```

### 3. 匹配素材

```bash
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/materials \
  -H "Content-Type: application/json" \
  -d '{"mode": "retrieval", "storyboard": {"shots": [...]}}'
```

### 4. 提交剪辑任务

```bash
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/edit \
  -H "Content-Type: application/json" \
  -d '{"storyboard": {...}, "material_selections": [...]}'
```

### 5. 查询进度

```bash
curl "http://localhost:8000/api/v1/projects/{project_id}/status?task_id={task_id}"
```

### 6. 获取结果

```bash
curl "http://localhost:8000/api/v1/projects/{project_id}/result?task_id={task_id}"
```

## 项目结构

```
├── app/
│   ├── agents/          # AI Agent 层 (导演 / 素材 / 剪辑)
│   ├── api/             # FastAPI 路由层
│   ├── core/            # 核心配置与基础设施 (数据库, LLM, Celery)
│   ├── models/          # SQLAlchemy ORM 模型
│   ├── services/        # 业务服务层 (文件, 索引, 渲染)
│   ├── workers/         # Celery 异步任务
│   └── main.py          # FastAPI 应用入口
├── tests/               # 单元测试与集成测试
├── alembic/             # 数据库迁移脚本
├── docs/                # 设计文档与报告
├── output/              # 渲染输出目录
├── docker-compose.yml   # 依赖服务编排
├── .env.example         # 环境变量模板
└── requirements.txt     # Python 依赖
```

## 开发指南

### 运行测试

```bash
# 运行全部测试
pytest

# 运行指定测试文件
pytest tests/test_api_integration.py -v

# 带覆盖率报告
pytest --cov=app tests/
```

### 代码风格

项目使用标准的 Python 类型注解和 FastAPI 最佳实践。提交前请确保：

1. 所有测试通过 (`pytest`)
2. 新增 API 端点包含对应的集成测试

### 添加新的 API 端点

1. 在 `app/api/` 下创建新的路由模块
2. 在 `app/main.py` 中注册路由
3. 在 `tests/` 下添加测试文件
4. 运行测试验证

## 设计文档

详见 [`docs/`](./docs) 目录，包含各模块的详细设计说明和任务报告。

## 许可证

MIT
