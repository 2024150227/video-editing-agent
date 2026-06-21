import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Seed sys.modules with fake modules BEFORE importing the module under test
for _mod_name in ("torch", "clip", "PIL", "PIL.Image"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from app.core.embedding import EmbeddingService


def _make_tensor_mock(data: np.ndarray):
    """Build a chain of MagicMock that emulates tensor.cpu().numpy()[0].tolist()."""
    m = MagicMock()
    m.norm.return_value = MagicMock()
    m.__truediv__.return_value = m  # features / norm -> same mock (no-op div)
    cpu_mock = MagicMock()
    cpu_mock.numpy.return_value = data
    m.cpu.return_value = cpu_mock
    return m


class TestEmbeddingService:
    def test_encode_text_returns_512d_vector(self):
        mock_model = MagicMock()
        mock_model.encode_text.return_value = _make_tensor_mock(
            np.random.randn(1, 512).astype(np.float32)
        )

        with patch.object(EmbeddingService, "_load_model"):
            with patch.object(EmbeddingService, "_tokenize") as mock_tok:
                mock_tok.return_value = MagicMock()
                service = EmbeddingService()
                service._model = mock_model
                service.device = "cpu"
                result = service.encode_text("一只猫在沙发上")
                assert isinstance(result, list)
                assert len(result) == 512

    @patch("PIL.Image.open")
    def test_encode_image_returns_512d_vector(self, mock_image_open):
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        mock_model = MagicMock()
        mock_model.encode_image.return_value = _make_tensor_mock(
            np.random.randn(1, 512).astype(np.float32)
        )
        mock_preprocess = MagicMock()
        mock_preprocess.return_value = MagicMock()

        with patch.object(EmbeddingService, "_load_model"):
            service = EmbeddingService()
            service._model = mock_model
            service._preprocess = mock_preprocess
            service.device = "cpu"
            result = service.encode_image("/fake/path.jpg")
            assert isinstance(result, list)
            assert len(result) == 512

    def test_encode_text_is_deterministic(self):
        mock_model = MagicMock()
        data = np.random.randn(1, 512).astype(np.float32)
        mock_model.encode_text.return_value = _make_tensor_mock(data)

        with patch.object(EmbeddingService, "_load_model"):
            with patch.object(EmbeddingService, "_tokenize") as mock_tok:
                mock_tok.return_value = MagicMock()
                service = EmbeddingService()
                service._model = mock_model
                service.device = "cpu"
                r1 = service.encode_text("test")
                r2 = service.encode_text("test")
                assert r1 == r2
