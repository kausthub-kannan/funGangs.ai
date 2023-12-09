from app.models.stylegan import generate_image
from app.database import add_image
import torch
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class GAN(BaseModel):
    user_id: str
    number_of_image: int
    alpha: float
    steps: int


@app.get("/")
def read_root():
    return "Welcome to FunGang generator! Generate humorous pixel arts of your gangs!"


@app.post("/pixel-art/generate")
def pixel_art_generation(gan: GAN):
    try:
        noise = torch.randn(gan.number_of_image, 256).to("cpu")
        generated_data = generate_image(noise, gan.alpha, gan.steps)
        data = add_image(gan.user_id, generated_data["url"])

        return data
    except RuntimeError:
        return {
            "statusCode": 500,
            "error": "MemoryError",
            "message": "Device Memory Overload: Please provide lesser number of images",
        }
