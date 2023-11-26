import torch
import requests
import os

from model.stylegan import generate_image
from database import supabase

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

        with open("generated_images/predicted.png", 'rb') as f:
            supabase.storage.from_("gangs_stylegan").upload(file=f, path="/", file_options={"content-type": "image/png"})

        if status == 200:
            return FileResponse("generated_images/predicted.png")
        else:
            return message
    except RuntimeError:
        return "Device Memory Overload: Please provide less number of images"


@app.get("/story/{payload}")
def get_story(payload: str):
    try:
        API_URL = "https://api-inference.huggingface.co/models/coffeeee/nsfw-story-generator2"
        headers = {"Authorization": os.getenv('HUGGING_FACE_KEY')}
        response = requests.post(API_URL, headers=headers, json=payload)
	    return response.json()
        
    except RuntimeError:
        return "API endpoint issue, please try again or try later"

