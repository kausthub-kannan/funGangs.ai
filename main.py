import torch
from stylegan import generate_image
import gradio as gr

noise = torch.randn(36, 256).to("cpu")


demo = gr.Interface(
    generate_image(noise),
    gr.Image(type="pil"),
    "image",
)

if __name__ == "__main__":
    demo.launch()
