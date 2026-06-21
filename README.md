# 基于多模态大模型的智能视频剪辑 Agent

基于多模态大模型（AgnesAPI）的智能视频剪辑系统。用户提供创意描述和素材，系统通过 **LangGraph** 编排三个 Agent 节点，自动完成分镜脚本生成、素材智能匹配、视频渲染的全流程。

## 系统架构

```
用户需求 ──► 导演节点 (LLM) ──► 分镜脚本 ──► [interrupt: 审核]
                                       │
                                       ▼
                                 素材节点 (CLIP) ──► 候选素材 ──► [interrupt: 选择]
                                       │
                                       ▼
                                 剪辑节点 (MoviePy) ──► 输出视频
```

| 组件 | 职责 |
|------|------|
| **导演节点** (`app/core/graph/nodes.py`) | 调用 AgnesAPI LLM 生成结构化分镜脚本 |
| **素材节点** (`app/core/graph/nodes.py`) | CLIP 语义检索 Qdrant 向量库 / AI 生成素材 |
| **剪辑节点** (`app/core/graph/nodes.py`) | MoviePy 时间线构建 + FFmpeg 同步渲染 |
| **向量检索** (`app/services/index_service.py`) | CLIP (ViT-B/32) 跨模态 embedding + Qdrant 向量搜索 |
| **渲染服务** (`app/services/render_service.py`) | MoviePy 时间线合成，支持图片/视频/背景音乐 |

## 技术栈

- **编排框架**: LangGraph + LangChain（带 checkpoint / interrupt / resume）
- **后端框架**: FastAPI (Python 3.11+)
- **多模态 LLM**: AgnesAPI (`agnes-2.0-flash`，免费，OpenAI 兼容)
- **向量数据库**: Qdrant + CLIP 嵌入 (ViT-B/32)
- **视频处理**: MoviePy 2.0+ / imageio-ffmpeg
- **数据库**: PostgreSQL 15 (SQLAlchemy 2.0 async)
- **测试**: pytest + httpx

## 快速启动

### 前置条件

- Python 3.11+
- Docker & Docker Compose
- AgnesAPI Key ([申请地址](https://platform.agnes-ai.com/)) — 免费

### 1. 启动依赖服务

```bash
docker-compose up -d
```

启动 PostgreSQL (5432)、Redis (6379)、Qdrant (6333)。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 `AGNES_API_KEY`。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

CLIP 模型首次运行时自动下载（~330MB），PyTorch 需手动安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. 数据库迁移

```bash
alembic upgrade head
```

### 5. 启动 API

```bash
uvicorn app.main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger 文档。

## API 工作流

```
POST /api/v1/projects                     → 创建项目
POST /api/v1/projects/{id}/story          → 生成分镜 → interrupt 暂停
POST /api/v1/projects/{id}/materials      → 匹配素材 → interrupt 暂停
POST /api/v1/projects/{id}/edit           → 渲染视频 → 同步返回结果
GET  /api/v1/projects/{id}/status         → 查询进度
GET  /api/v1/projects/{id}/result         → 下载视频
```

### 使用示例

```bash
# 1. 创建项目
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "旅行vlog", "material_source_dir": "./test_materials"}'

# 2. 生成分镜（LLM 生成后暂停，返回待审核的分镜）
curl -X POST http://localhost:8000/api/v1/projects/{id}/story \
  -H "Content-Type: application/json" \
  -d '{"prompt": "15秒海边旅行快剪", "style": "快节奏", "total_duration_sec": 15}'

# 3. 确认/修改分镜并匹配素材（CLIP 检索后暂停）
curl -X POST http://localhost:8000/api/v1/projects/{id}/materials \
  -H "Content-Type: application/json" \
  -d '{"mode": "retrieval", "storyboard": {"shots": [...]}}'

# 4. 选择素材并渲染（同步输出视频）
curl -X POST http://localhost:8000/api/v1/projects/{id}/edit \
  -H "Content-Type: application/json" \
  -d '{"storyboard": {...}, "material_selections": [{"shot_index":1,"file_path":"..."}]}'
```

## LangGraph 状态流转

```
START → director_node → [interrupt: 分镜审核]
      → material_node → [interrupt: 素材选择]
      → editor_node → END
```

- `graph.ainvoke(initial_state, config)` — 启动，遇 `interrupt()` 暂停
- `graph.ainvoke(Command(resume=value), config)` — 继续执行
- `config = {"configurable": {"thread_id": project_id}}` — 用 project_id 做 checkpoint key
- MemorySaver 在内存中保存状态，进程重启后丢失

## 项目结构

```
├── app/
│   ├── api/                  # FastAPI 路由 (projects / story / materials / edit)
│   ├── core/
│   │   ├── graph/            # LangGraph: state / nodes / tools / llm / builder
│   │   ├── config.py         # pydantic-settings 配置
│   │   ├── database.py       # SQLAlchemy async engine
│   │   ├── embedding.py      # CLIP 向量化服务
│   │   └── celery_app.py     # Celery 配置 (保留，渲染已改为同步)
│   ├── agents/               # Agent 层 (兼容层，核心逻辑在 graph/)
│   ├── services/             # 业务服务 (索引 / 文件 / 渲染)
│   ├── models/               # SQLAlchemy ORM
│   ├── workers/              # Celery Worker (保留异步能力)
│   └── main.py               # FastAPI 入口
├── tests/                    # pytest 单元/集成测试
├── alembic/                  # 数据库迁移
├── test_materials/           # 测试用素材
├── output/                   # 渲染输出目录
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 集成测试
pytest tests/test_api_integration.py -v

# 端到端渲染测试
pytest tests/test_celery_render.py -v -s
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `No module named 'pkg_resources'` | 新版 setuptools 移除了该模块，已通过 monkey-patch 修复 |
| `context length 77` | CLIP tokenizer 上限，已设置 `truncate=True` |
| FFmpeg 管道写入失败 | Windows + Celery 兼容性问题，渲染已改为同步执行 |
| PostgreSQL 连接拒绝 | `docker-compose up -d` 启动数据库 |

## 设计文档

详见 [`docs/`](./docs) 目录和 `CLAUDE.md`。

## 许可证

MIT
