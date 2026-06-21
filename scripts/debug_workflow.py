"""调试脚本 — 端到端测试视频剪辑 Agent 完整流程。

使用方式:
    python scripts/debug_workflow.py

前提:
    1. docker-compose up -d   (PostgreSQL + Redis + Qdrant)
    2. uvicorn app.main:app --port 8000
    3. celery -A app.core.celery_app worker -c 1
    4. 配置好 .env 中的 VOLCANO_ARK_API_KEY
"""

import requests
import time
import sys

BASE = "http://localhost:8000/api/v1"

# ─── 检查服务是否就绪 ───
try:
    r = requests.get("http://localhost:8000/health", timeout=3)
    r.raise_for_status()
    print(f"✅ API 服务就绪 — {r.json()}")
except Exception as e:
    print(f"❌ API 服务未启动: {e}")
    print("请先运行: uvicorn app.main:app --port 8000")
    sys.exit(1)


def step(msg, method, path, **kwargs):
    """执行一步并打印结果"""
    print(f"\n{'='*60}")
    print(f"📌 {msg}")
    url = f"{BASE}{path}"
    resp = method(url, **kwargs)
    print(f"   Status: {resp.status_code}")
    try:
        data = resp.json()
        # 截断过长的输出
        text = str(data)
        if len(text) > 500:
            text = text[:500] + "...(截断)"
        print(f"   Body: {text}")
        return data
    except Exception:
        print(f"   Body: {resp.text[:300]}")
        return resp.text


# ─── Step 1: 创建项目 ───
project = step(
    "创建项目",
    requests.post,
    "/projects",
    json={
        "name": "调试测试-旅行vlog",
        "material_source_dir": "./test_materials",  # 改成你的素材目录
        "aspect_ratio": "9:16",
        "resolution": "1080x1920",
    },
)
project_id = project.get("project_id", "proj-test")


# ─── Step 2: 生成分镜 ───
story = step(
    "导演Agent — 生成分镜脚本",
    requests.post,
    f"/projects/{project_id}/story",
    json={
        "prompt": "旅行vlog快节奏卡点短片，记录海边的美好时光",
        "style": "快节奏卡点",
        "total_duration_sec": 15,
    },
)
print("\n⚠️  此时 LangGraph 已在 director_node 处 interrupt")
print("   请检查上面的分镜脚本，确认后继续...")


# ─── Step 3: 匹配素材 ───
# 如果分镜里有 shots，就用它；否则跳过
if story and "shots" in story:
    materials = step(
        "素材Agent — 检索匹配素材",
        requests.post,
        f"/projects/{project_id}/materials",
        json={
            "mode": "retrieval",       # 或 "generation"
            "storyboard": story,
        },
    )
    print("\n⚠️  此时 LangGraph 已在 material_node 处 interrupt")
    print("   请检查上面的候选素材，确认后继续...")

    # ─── Step 4: 提交剪辑 ───
    edit = step(
        "剪辑Agent — 提交渲染",
        requests.post,
        f"/projects/{project_id}/edit",
        json={
            "storyboard": story,
            "material_selections": [
                {"shot_index": s["index"], "file_path": "/path/to/default.mp4"}
                for s in story.get("shots", [])
            ],
        },
    )

    # ─── Step 5: 轮询状态 ───
    task_id = edit.get("task_id")
    if task_id:
        print(f"\n⏳ 等待渲染完成... (task_id: {task_id})")
        for i in range(60):  # 最多等 2 分钟
            status = step(
                f"查询状态 ({i+1}/60)",
                requests.get,
                f"/projects/{project_id}/status",
                params={"task_id": task_id},
            )
            if status.get("status") == "completed":
                print("\n🎬 渲染完成！")
                result = requests.get(
                    f"{BASE}/projects/{project_id}/result",
                    params={"task_id": task_id},
                ).json()
                print(f"   视频路径: {result.get('file_path')}")
                print(f"   文件大小: {result.get('file_size_bytes', 0) / 1024 / 1024:.1f} MB")
                break
            elif status.get("status") == "failed":
                print(f"\n❌ 渲染失败: {status.get('error')}")
                break
            time.sleep(2)
else:
    print("\n⚠️  分镜生成失败（可能是火山方舟 API Key 未配置），跳过后续步骤")

print(f"\n{'='*60}")
print("🏁 调试流程结束")
