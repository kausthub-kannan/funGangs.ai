from server.models.stylegan import generate_image
from server.database import add_image
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import FileResponse


app = FastAPI()


class GAN(BaseModel):
    user_id: str | None = "46d7b008-92d8-11ee-b9d1-0242ac120002"
    number_of_image: int | None = 8
    alpha: float | None = 1
    steps: int | None = 5


@app.get("/")
def read_root():
    return "Welcome to FunGang generator! Generate humorous pixel arts of your gangs!"


@app.post("/generate")
def image_generation(gan: GAN):
    try:
        noise = torch.randn(gan.number_of_image, 256).to("cpu")
        status, message1 = generate_image(noise, gan.alpha, gan.steps)
        status, message2 = add_image(gan.user_id, "generated_images/predicted.png")

        if status == 200:
            return FileResponse("generated_images/predicted.png")
        else:
            return {message1, message2}
    except RuntimeError:
        return "Device Memory Overload: Please provide lesser number of images"

