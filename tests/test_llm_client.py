import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.llm_client import LLMClient


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_chat_returns_text_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "你好，世界"}}]
        }

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat([{"role": "user", "content": "你好"}])
            assert result == "你好，世界"

    @pytest.mark.asyncio
    async def test_chat_json_parses_valid_json(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}]
        }

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "返回JSON"}])
            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_chat_json_retries_on_invalid_json(self):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.status_code = 200
            if call_count < 3:
                mock.json.return_value = {"choices": [{"message": {"content": "not json {"}}]}
            else:
                mock.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
            return mock

        with patch.object(httpx.AsyncClient, "post", side_effect=side_effect):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "JSON please"}], retries=3)
            assert result == {"ok": True}
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_chat_json_returns_raw_on_exhausted_retries(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "totally broken"}}]}

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "JSON please"}], retries=2)
            assert result == {"raw": "totally broken", "error": "JSON parse failed after 2 retries"}
