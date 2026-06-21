import logging
from functools import lru_cache
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """CLIP 跨模态 Embedding 服务 — 文本和图像映射到同一向量空间"""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        settings = get_settings()
        self.model_name = model_name or settings.CLIP_MODEL
        self.device = device if self._is_cuda_available() else "cpu"
        self._model, self._preprocess = None, None
        self._load_model()

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        import clip
        import torch
        logger.info(f"Loading CLIP model '{self.model_name}' on {self.device}...")
        self._model, self._preprocess = clip.load(self.model_name, device=self.device)
        self._model.eval()

    @staticmethod
    def _tokenize(text: str):
        import clip
        return clip.tokenize([text])

    def encode_text(self, text: str) -> list[float]:
        """将文本编码为向量"""
        import torch
        tokenized = self._tokenize(text).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokenized)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def encode_image(self, image_path: str) -> list[float]:
        """将图片编码为向量"""
        import torch
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()
