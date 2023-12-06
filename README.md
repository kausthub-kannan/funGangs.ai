# funGangs.ai

![alt text](https://i.ibb.co/98LQsXM/Screenshot-20231013-154312.png)

## NOTE!! 
The server is down due to an increase in model weight size
But the image can be found at [DockerHub](https://hub.docker.com/repository/docker/kausthubkannan/fungangs-ai)

## Background of the Model
1. The model is trained on `StyleGAN` and PyTorch is used for the development.  
2. The Jupyter Notebook is present in the following Kaggle [link](https://www.kaggle.com/code/kausthubkannan/character-generation-stylegan).  
3. The dataset which is trained upon is `Pixelated Treasures: 10K CryptoPunks` 
which can be found via this [link](https://www.kaggle.com/datasets/chwasiq0569/cryptopunks-pixel-art-dataset/).

## Features
#### Image Generation:
The image generation as said uses a StyleGAN custom-trained model.

#### Story Generation: (TODO)
Currently, the story generation is supported by this [Hugging Face API interface](https://api-inference.huggingface.co/models/coffeeee/nsfw-story-generator2). For further addition and gamification, custom trained LLM model will be added.

#### Database Support:
The future of this project aims to gamify and make it user-specific. Hence, Database support has been added and is in progress to store user-specific data and images.

## ToDo
The project further aims to expand to below features:
- [x] FunGangs Image Generation
- [ ] Story Generation based on Image-context
- [ ] User Interactive Pix2Pix GAN Image Generation
- [ ] Wiki API for the Gangs character

## API Docs
Currently, the APIs support the generation of images and future expansion is the integration of LLMs for story generations.

### Endpoints

**Image Generation:** `/generated`  
The endpoint requires number_of_images to be entered. The response would be a single image which is a collage of a given 
number of images as shown in the image.  
  
Example:  

**1. Fungangs Image Generation:**

**JavaScript**
```javascript
import axios from "axios"

try{
  const number_of_images = 16
  const url = "http://127.0.0.1.8080/generated/{number_of_images}"
  const body = {"user_id": valid uuid4, "number_of_images":8}
  const generated_image = axios.get(url, body)
}catch(err){
  console.log(err)
}
```

**Python**  
```python
import requests

try:
    number_of_images = 16
    url = f"http://127.0.0.1.8080/generated"
    body = {"user_id": valid uuid4, "number_of_images":8}
    response = requests.get(url, json=body)
    generated_image = response.content
except for Exception as err:
    print(err)
```

### Note
The model, which is based on **StyleGAN**, requires high memory in terms of CPU and CUDA. Hence, please `restrain from generating a number of images more than the range of (8, 48)`. The story generation is on BETA as it is based on the existing Hugging Face interface. Soon custom LLM support will be added."




