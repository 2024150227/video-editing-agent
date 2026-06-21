# CLAUDE.md — 智能视频剪辑 Agent

## 项目概述

基于多模态大模型（火山方舟）的智能视频剪辑 API 服务，使用 LangGraph 编排三个 Agent：
- **导演 Agent** — 创意理解 → 分镜脚本生成
- **素材 Agent** — 跨模态检索（CLIP + Qdrant）或 AI 生成
- **剪辑 Agent** — MoviePy 时间线构建 + FFmpeg 渲染

## 技术栈

- **框架**: FastAPI + LangGraph + LangChain
- **模型**: 火山方舟 API (OpenAI 兼容，通过 ChatOpenAI 适配)
- **向量**: CLIP (ViT-B/32) + Qdrant
- **视频**: MoviePy 2.0+ + FFmpeg
- **任务**: Celery + Redis
- **数据库**: PostgreSQL + SQLAlchemy 2.0 (async)
- **Python**: 3.11+

## 项目结构

```
app/
├── main.py              # FastAPI 入口
├── api/                 # REST 端点（依赖 LangGraph）
│   ├── projects.py      # 项目 CRUD
│   ├── story.py         # 导演节点 → interrupt
│   ├── materials.py     # 素材节点 → interrupt
│   └── edit.py          # 剪辑节点 → 渲染
├── core/
│   ├── graph/           # LangGraph 核心
│   │   ├── state.py     # VideoEditingState
│   │   ├── nodes.py     # 3 个节点函数
│   │   ├── tools.py     # search_materials, generate_material
│   │   ├── llm.py       # ChatOpenAI 工厂（火山方舟适配）
│   │   └── builder.py   # StateGraph 构建 + MemorySaver
│   ├── config.py        # pydantic-settings
│   ├── database.py      # SQLAlchemy async engine
│   ├── celery_app.py    # Celery 配置
│   ├── embedding.py     # CLIP 向量化
│   └── llm_client.py    # 原 LLM 客户端（已被 LangChain 替代）
├── agents/              # 原 3 Agent 类（保留兼容）
├── services/
│   ├── index_service.py # Qdrant 向量索引
│   ├── file_service.py  # 文件扫描/关键帧提取
│   └── render_service.py# MoviePy 渲染
├── models/              # SQLAlchemy ORM
└── workers/
    ├── render_worker.py # 渲染 Celery Task
    └── index_worker.py  # 索引 Celery Task
```

## 快速启动

```bash
docker-compose up -d                          # PostgreSQL + Redis + Qdrant
cp .env.example .env                          # 配置火山方舟 API Key
alembic upgrade head                          # 数据库迁移
uvicorn app.main:app --reload --port 8000     # API
celery -A app.core.celery_app worker -c 1     # Worker
```

## API 工作流

```
POST /projects              → 创建项目
POST /{id}/story            → graph.ainvoke() → interrupt(分镜) → 返回分镜
POST /{id}/materials        → Command(resume=分镜) → interrupt(素材) → 返回候选
POST /{id}/edit             → Command(resume=素材) → 渲染 → 完成后返回
GET  /{id}/status?task_id=  → 查询进度
GET  /{id}/result?task_id=  → 下载视频
```

## LangGraph 状态流转

```
START → director_node → [interrupt: 分镜审核]
      → material_node → [interrupt: 素材选择]
      → editor_node → END
```

- `graph.ainvoke(initial_state, config)` — 启动，遇 interrupt 暂停
- `graph.ainvoke(Command(resume=value), config)` — 继续执行
- `config = {"configurable": {"thread_id": project_id}}` — 用 project_id 做 checkpoint key

## 测试

```bash
pytest tests/ -v          # 60 tests
pytest tests/test_api_integration.py -v  # 仅集成测试
```

## 注意事项

- 火山方舟 API 是 OpenAI 兼容的，base_url 指向 `https://ark.cn-beijing.volces.com/api/v3`
- CLIP 模型首次加载需下载 ~330MB，后续有 lru_cache
- 视频渲染是 CPU 密集型，Celery worker 建议 concurrency=1
- 素材索引是异步任务，需启动 Celery Worker 后才执行
