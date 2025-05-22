# tests/test_main.py
import pytest
from httpx import AsyncClient
from app.main import app # Assuming your FastAPI app instance is named 'app' in app/main.py

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}