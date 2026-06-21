# 基于多模态大模型的智能视频剪辑 Agent — 设计文档

**日期**: 2026-06-21
**状态**: 设计阶段

---

## 一、项目概述

构建一个基于多模态大模型的智能视频剪辑 API 服务。用户通过自然语言描述创意意图，Agent 自动完成分镜拆解、跨模态素材匹配、视频剪辑合成三步流程，每一步支持人工干预和调整。

### 核心能力

- **文生素材**：通过文本描述直接生成图片或视频片段
- **智能剪辑**：从用户自有素材库中跨模态检索匹配，自动剪辑成片

---

## 二、目标用户

个人内容创作者 / 自媒体博主，拥有大量拍摄素材，需要快速将创意转化为短视频。

---

## 三、系统架构

```
                          ┌──────────────────┐
                          │   API Gateway     │
                          │   (FastAPI)       │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │  /story   │ │ /materials│ │  /edit    │
              │  导演Agent │ │ 素材Agent │ │ 剪辑Agent │
              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                    │              │              │
         ┌─────────┼──────────────┼──────────────┼─────────┐
         │         │              │              │         │
    ┌────▼────┐ ┌──▼───┐  ┌──────▼──────┐ ┌─────▼────┐   │
    │火山方舟  │ │向量库 │  │ 素材索引服务 │ │ FFmpeg   │   │
    │ LLM API │ │(Qdrant│  │ (文件扫描,   │ │ MoviePy  │   │
    │         │ │ )     │  │  特征提取)  │ │ 渲染引擎 │   │
    └─────────┘ └──────┘  └─────────────┘ └──────────┘   │
         │                                                   │
         └───────────────────────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │ PostgreSQL│
                    │ + Redis   │
                    │ (状态/任务)│
                    └───────────┘
```

### 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| API 框架 | FastAPI | 异步支持好，自动文档生成 |
| 多模态模型 | 火山方舟 API | 视觉理解 + 文本生成 |
| 向量数据库 | Qdrant | 开源，支持多模态 Embedding |
| Embedding | CLIP (OpenAI) | 文本-图像跨模态对齐 |
| 视频处理 | MoviePy + FFmpeg | Python 原生操作 + 命令行渲染 |
| 异步任务 | Celery + Redis | 异步渲染，进度上报 |
| 数据库 | PostgreSQL | 项目/任务持久化 |
| 语言 | Python 3.11+ | AI/ML 生态最成熟 |

---

## 四、多 Agent 设计

### Agent 1：导演 Agent（Director）

**职责**：创意理解 + 分镜拆解

**输入**：用户自然语言描述
**输出**：结构化分镜脚本（Storyboard JSON）

**内部流程**：

```
用户文本 → 意图理解(LLM补全风格/时长/节奏)
         → 分镜生成(LLM JSON Mode, 秒级拆解)
         → 规则校验(时长/节奏/转场逻辑检查)
         → Storyboard JSON
```

**设计要点**：
- 两步 LLM 调用：先意图增强再分镜生成，质量优于一步直出
- 使用火山方舟 Function Calling 强制结构化输出
- 规则引擎兜底：时长不匹配、转场不合理时自动 retry

### Agent 2：素材 Agent（Material）

**职责**：跨模态检索 + 生成

**输入**：分镜描述文本 + 模式选择
**输出**：每镜 3-5 个候选素材 + 置信度 + 理由

**模式一 — 智能检索**：

```
用户素材目录 → 后台异步索引(关键帧抽取 → CLIP Embedding → Qdrant)
查询时：描述文本 → CLIP文本编码 → Qdrant相似度检索(Top-10)
        → LLM精排(语义理解+裁剪最佳片段) → MaterialMatch[]
```

**模式二 — AI 生成**：

```
描述文本 → LLM扩写Prompt(补充构图/光影/风格)
        → 火山方舟文生图/视频 API → 生成素材
```

**设计要点**：
- 素材入库是预构建的异步任务，检索时毫秒级响应
- 粗排（向量检索）+ 精排（LLM 语义确认）两级搜索
- AI 生成前 LLM 需要扩写 Prompt，用户原始描述往往过于简略

### Agent 3：剪辑 Agent（Editor）

**职责**：时间线编排 + 渲染导出

**输入**：确认后的分镜 + 选定的素材列表
**输出**：最终视频文件

**内部流程**：

```
分镜+素材 → 时间线构建(裁剪/排序/转场)
          → 画面调整(分辨率统一/色彩一致性)
          → 音频处理(BGM/音量/卡点对齐)
          → FFmpeg渲染合成
          → 质量检查(时长/同步/格式)
          → 返回视频URL
```

**设计要点**：
- 异步任务模型（Celery）：请求立即返回 task_id，客户端轮询进度
- MoviePy 做时间线编排和简单合成，FFmpeg CLI 做最终高质量渲染
- 字幕叠加用 MoviePy TextClip + PIL，需指定中文字体路径

---

## 五、API 设计

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/projects` | 创建剪辑项目 |
| POST | `/api/v1/projects/{id}/story` | 导演Agent：生成分镜 |
| POST | `/api/v1/projects/{id}/materials` | 素材Agent：检索匹配 |
| POST | `/api/v1/projects/{id}/edit` | 剪辑Agent：渲染合成（异步） |
| GET | `/api/v1/projects/{id}/status` | 查询任务进度 |
| GET | `/api/v1/projects/{id}/result` | 获取最终视频 |

### 请求/响应示例

**POST /projects** — 创建项目：
```json
// Request
{
  "name": "旅行vlog-2024巴厘岛",
  "material_source_dir": "/data/materials/bali_2024/",
  "setting": {
    "aspect_ratio": "9:16",
    "resolution": "1080x1920"
  }
}
// Response
{ "project_id": "proj_abc123", "name": "...", "created_at": "..." }
```

**POST /projects/{id}/story** — 生成分镜：
```json
// Request
{
  "prompt": "把旅行素材剪成30秒快节奏卡点视频，配vlog风格",
  "style": "快节奏卡点",
  "total_duration_sec": 30
}
// Response: Storyboard JSON (见数据模型)
```

**POST /projects/{id}/materials** — 检索素材：
```json
// Request
{
  "mode": "retrieval",        // "retrieval" | "generation"
  "storyboard": { ... },      // 确认后的分镜脚本
  "material_overrides": {     // 用户可指定某镜用特定文件
    "1": "MVI_8843.mp4"
  }
}
// Response: MaterialMatch[] (见数据模型)
```

**POST /projects/{id}/edit** — 开始剪辑：
```json
// Request
{
  "storyboard": { ... },      // 最终确认的分镜
  "material_selections": [    // 用户每镜选定的素材
    { "shot_index": 1, "material_id": "m_001" },
    { "shot_index": 2, "material_id": "m_005" }
  ],
  "bgm_path": "/data/music/upbeat.mp3",  // 可选
  "subtitles": "自动生成"     // 可选
}
// Response
{ "task_id": "task_abc123", "status": "queued" }
```

### 创作流程

```
客户端                         服务端
  │                              │
  ├─ POST /projects ────────────→ 创建项目，返回 project_id
  │                              │
  ├─ POST /{id}/story ──────────→ 导演Agent → 返回 Storyboard
  │                              │
  ├─ 用户确认/调整分镜             │
  │                              │
  ├─ POST /{id}/materials ──────→ 素材Agent → 返回候选素材
  │                              │
  ├─ 用户确认/替换素材             │
  │                              │
  ├─ POST /{id}/edit ───────────→ 创建异步任务，返回 task_id
  │  GET /{id}/status ←────────── 轮询进度
  │  GET /{id}/result ←────────── 下载视频
```

---

## 六、数据模型

### Project（项目）

```json
{
  "project_id": "proj_abc123",
  "name": "旅行vlog-2024巴厘岛",
  "material_source_dir": "/data/materials/bali_2024/",
  "setting": {
    "aspect_ratio": "9:16",
    "resolution": "1080x1920"
  },
  "index_status": "ready",       // "pending" | "indexing" | "ready" | "failed"
  "created_at": "2026-06-21T10:00:00Z"
}
```

### Storyboard（分镜脚本）

```json
{
  "project_id": "proj_xxx",
  "style": "快节奏卡点",
  "total_duration_sec": 30,
  "bgm_mood": "电子/动感",
  "shots": [
    {
      "index": 1,
      "duration_sec": 3,
      "description": "飞机起飞升空，云层掠过，大远景",
      "shot_type": "wide",
      "transition": "硬切",
      "mood_words": ["自由", "启程"]
    }
  ]
}
```

### MaterialMatch（素材匹配）

```json
{
  "shot_index": 1,
  "candidates": [
    {
      "material_id": "m_001",
      "file_name": "MVI_8843.mp4",
      "clip_range": [12.0, 16.0],
      "score": 0.92,
      "reason": "画面包含天空+云层，与'飞机起飞'描述高度匹配"
    }
  ]
}
```

### EditTask（剪辑任务）

```json
{
  "task_id": "task_abc123",
  "project_id": "proj_xxx",
  "status": "rendering",
  "progress_pct": 45,
  "estimated_remaining_sec": 120,
  "output_url": null
}
```

---

## 七、项目目录结构

```
video-editing-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── projects.py          # 项目 CRUD 端点
│   │   ├── story.py             # 导演 Agent 端点
│   │   ├── materials.py         # 素材 Agent 端点
│   │   └── edit.py              # 剪辑 Agent 端点
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── director.py          # 导演 Agent
│   │   ├── material.py          # 素材 Agent
│   │   └── editor.py            # 剪辑 Agent
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # 配置管理
│   │   ├── llm_client.py        # 火山方舟 LLM 客户端
│   │   └── embedding.py         # CLIP Embedding 服务
│   ├── services/
│   │   ├── __init__.py
│   │   ├── index_service.py     # 素材索引服务
│   │   ├── render_service.py    # 视频渲染服务（Celery Worker）
│   │   └── file_service.py      # 文件管理服务
│   ├── models/
│   │   ├── __init__.py
│   │   ├── project.py           # Project ORM
│   │   ├── storyboard.py        # Storyboard / Shot schema
│   │   ├── material.py          # Material schema
│   │   └── task.py              # EditTask ORM
│   └── workers/
│       ├── __init__.py
│       └── render_worker.py     # Celery Worker
├── tests/
│   ├── test_director.py
│   ├── test_material.py
│   └── test_editor.py
├── alembic/                     # 数据库迁移
├── requirements.txt
├── docker-compose.yml           # PostgreSQL + Redis + Qdrant
└── README.md
```

---

## 八、错误处理策略

| 场景 | 策略 |
|------|------|
| LLM 输出格式异常 | JSON 解析失败 → 自动 retry × 3，仍失败返回原始输出 + 错误提示 |
| 素材检索无结果 | 返回空候选列表 + 建议：降低阈值 / 切换到 AI 生成模式 |
| 视频渲染失败 | Celery 任务标记 failed，错误信息持久化，支持重试 |
| 素材文件不存在 | 索引阶段校验，渲染前二次校验，缺失时标记并跳过该镜头 |
| 并发限制 | 单用户同时最多 1 个渲染任务，API 层返回 429 |

---

## 九、测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | 分镜解析逻辑、素材匹配排序、时间线构建 | pytest |
| Agent 集成测试 | 导演→素材→剪辑 完整流程，Mock LLM 调用 | pytest + fixtures |
| API 测试 | 端点输入输出校验、异常路径 | httpx / TestClient |
| LLM 质量测试 | 固定输入→人工评估分镜质量、素材匹配准确率 | 离线评测集 |

---

## 十、后续扩展方向

- 字幕自动生成（ASR → 文案 → 双语字幕）
- 多镜头同步（多机位同时拍摄素材的自动对齐）
- 风格迁移（将参考视频的风格应用到输出视频）
- 批量生产模式（单次生成 A/B/C 多个版本供选择）
- Web UI 管理后台

---

## 十一、关键开源依赖

| 依赖 | 用途 | 版本建议 |
|------|------|----------|
| fastapi | API 框架 | ≥0.110 |
| moviepy | 视频时间线编排 | ≥2.0 |
| celery | 异步任务队列 | ≥5.3 |
| qdrant-client | 向量数据库客户端 | ≥1.9 |
| openai-clip | 跨模态 Embedding | 最新 |
| sqlalchemy | ORM | ≥2.0 |
| alembic | 数据库迁移 | ≥1.13 |
| redis | 缓存 + 任务队列 Backend | ≥7.0 |
| ffmpeg-python | FFmpeg Python 绑定 | 最新 |
| librosa | 音频节拍检测 | ≥0.10 |
| pillow | 字幕图片生成 | ≥10.0 |
| uvicorn | ASGI 服务器 | ≥0.27 |
