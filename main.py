import torch
from stylegan import generate_image
from fastapi import FastAPI
from fastapi.responses import FileResponse


app = FastAPI()


@app.get("/")
def read_root():
    return "Welcome to FunGang generator! Generate humorous pixel arts of your gangs!"


@app.get("/generate/{number_of_image}")
def image_generation(number_of_image: int):
    try:
        noise = torch.randn(number_of_image, 256).to("cpu")
        status, message = generate_image(noise)

        if status == 200:
            return FileResponse("generated_images/predicted.png")
        else:
            return message
    except RuntimeError:
        return "Device Memory Overload: Please provide less number of images"
