import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def settings_override():
    from app.core.config import Settings
    return Settings(
        VOLCANO_ARK_API_KEY="test_key",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        DATABASE_URL_SYNC="sqlite:///:memory:",
        OUTPUT_DIR="./test_output",
    )
