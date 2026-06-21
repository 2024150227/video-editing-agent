import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import get_settings
from app.core.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class IndexService:
    """素材索引服务 — 管理 Qdrant 向量数据库中的素材"""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
    ):
        settings = get_settings()
        self.embedding = embedding_service or EmbeddingService()
        self.client = qdrant_client or QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
        self.collection = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    def index_single(self, project_id: str, file_info: dict, keyframe_path: str) -> str:
        """索引单个素材（图片或视频关键帧）"""
        vector = self.embedding.encode_image(keyframe_path)
        point_id = f"{project_id}_{file_info['file_name']}_{hash(file_info['file_path'])}"

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "project_id": project_id,
                        "file_name": file_info["file_name"],
                        "file_path": file_info["file_path"],
                        "file_type": file_info["file_type"],
                    },
                )
            ],
        )
        return point_id

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """向量相似度检索，返回 Top-K 结果"""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {
                "point_id": r.id,
                "score": round(r.score, 4),
                "file_path": r.payload.get("file_path", ""),
                "file_name": r.payload.get("file_name", ""),
                "file_type": r.payload.get("file_type", ""),
                "project_id": r.payload.get("project_id", ""),
            }
            for r in results
        ]

    def search_by_text(self, text: str, top_k: int = 10) -> list[dict]:
        """用文本描述搜索素材"""
        vector = self.embedding.encode_text(text)
        return self.search(vector, top_k)

    def delete_project(self, project_id: str):
        """删除项目关联的所有向量"""
        self.client.delete(
            collection_name=self.collection,
            points_selector={"filter": {"must": [{"key": "project_id", "match": {"value": project_id}}]}},
        )
