"""测试 Celery Worker + MoviePy 渲染端到端流程"""
import os
import time
import pytest
import requests

BASE_URL = "http://localhost:8004"
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCeleryRenderFullFlow:
    """端到端测试：创建项目 → 分镜 → 编辑渲染"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.beach = os.path.join(PROJ_DIR, "test_materials", "beach.jpg")
        self.sunset = os.path.join(PROJ_DIR, "test_materials", "sunset.jpg")
        if not os.path.exists(self.beach) or not os.path.exists(self.sunset):
            pytest.skip("Test material images not found")

    def test_full_render_flow(self):
        """完整的 API → Celery 渲染流程"""
        # 1. 创建项目
        r = requests.post(f"{BASE_URL}/api/v1/projects", json={
            "name": "celery-test", "material_source_dir": "./test_materials"
        })
        assert r.status_code == 201, f"Create failed: {r.text}"
        pid = r.json()["project_id"]
        print(f"Project: {pid}")

        # 2. 生成分镜
        r = requests.post(f"{BASE_URL}/api/v1/projects/{pid}/story", json={
            "prompt": "5秒旅行快剪", "style": "快节奏", "total_duration_sec": 5,
        }, timeout=120)
        assert r.status_code == 200, f"Story failed: {r.text[:300]}"
        print(f"Story: {len(r.json()['shots'])} shots")

        # 3. 提交编辑（Material selections resume）
        r = requests.post(f"{BASE_URL}/api/v1/projects/{pid}/edit", json={
            "storyboard": {
                "style": "fast", "total_duration_sec": 5,
                "shots": [
                    {"index": 1, "duration_sec": 2, "description": "beach",
                     "shot_type": "wide", "transition": "hard cut"},
                    {"index": 2, "duration_sec": 3, "description": "sunset",
                     "shot_type": "medium", "transition": "fade"},
                ]
            },
            "material_selections": [
                {"shot_index": 1, "file_path": self.beach},
                {"shot_index": 2, "file_path": self.sunset},
            ],
        }, timeout=180)
        assert r.status_code == 200, f"Edit failed: {r.text[:500]}"
        data = r.json()
        task_id = data.get("task_id")
        print(f"Edit: task_id={task_id}, status={data.get('status')}")

        # task_id 可能为 null(如果 Celery 未运行)，跳过后续轮询
        if not task_id:
            pytest.skip("Celery worker not running — task_id is null")

        # 4. 轮询状态直到完成
        for i in range(30):
            time.sleep(2)
            r = requests.get(f"{BASE_URL}/api/v1/projects/{pid}/status",
                             params={"task_id": task_id})
            s = r.json()
            status = s["status"]
            pct = s["progress_pct"]
            print(f"  [{i}] {status:12s} {pct}%")

            if status == "completed":
                output_path = s.get("output_path", "")
                assert os.path.exists(output_path), f"Output not found: {output_path}"
                size = os.path.getsize(output_path)
                print(f"  Output: {output_path} ({size} bytes)")
                assert size > 1000, f"Output file too small: {size} bytes"
                return

            if status == "failed":
                pytest.fail(f"Render failed: {s.get('error', 'Unknown')}")

        pytest.fail("Render did not complete within 60 seconds")
