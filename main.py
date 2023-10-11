import torch
from stylegan import Generator

model = Generator(Z_DIM, W_DIM, IN_CHANNELS)
model = torch.load("/workspace/mafia-gangs/models/generator.pt", map_location=torch.device('cpu'))

noise = torch.randn(32, 256).to("cpu")
steps = 5
alpha = 1
model.eval()
img = model(noise, alpha, steps)
save_image(img*0.5+0.5, f"test_img.png")