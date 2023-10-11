import torch
from stylegan import generate_image

noise = torch.randn(32, 256).to("cpu")

import gradio as gr
import os


demo = gr.Interface(
    generate_image(noise),
    gr.Image(type="pil"),
    "image",
)

if __name__ == "__main__":
    demo.launch()
