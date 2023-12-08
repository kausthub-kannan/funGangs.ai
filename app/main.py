from app.models.stylegan import generate_image
from app.database import add_image
import torch
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class GAN(BaseModel):
    user_id: str | None = "46d7b008-92d8-11ee-b9d1-0242ac120002"
    number_of_image: int | None = 1
    alpha: float | None = 1
    steps: int | None = 5


@app.get("/")
def read_root():
    return "Welcome to FunGang generator! Generate humorous pixel arts of your gangs!"


@app.post("/pixel-art/generate")
def image_generation(gan: GAN):
    try:
        noise = torch.randn(gan.number_of_image, 256).to("cpu")
        url, _ = generate_image(noise, gan.alpha, gan.steps)
        data = add_image(gan.user_id, url)
        return data
    except RuntimeError:
        return {
            "statusCode": 500,
            "error": "MemoryError",
            "message": "Device Memory Overload: Please provide lesser number of images",
        }
