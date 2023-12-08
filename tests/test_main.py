from fastapi.testclient import TestClient
from pydantic import BaseModel
from app.main import app

client = TestClient(app)


class PixelArtGenerate(BaseModel):
    user_id: str = "46d7b008-92d8-11ee-b9d1-0242ac120002"
    number_of_image: int = 1
    alpha: float = 1.0
    steps: int = 5


def pixel_art_test():
    response = client.post(
        "/pixel-art/generate", json=PixelArtGenerate().model_dump_json()
    )
    assert response.status_code == 200
