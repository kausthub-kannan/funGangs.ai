# funGangs.ai

![alt text](https://i.ibb.co/98LQsXM/Screenshot-20231013-154312.png)

## Background of the Model
1. The model is trained on `StyleGAN` and PyTorch is used for the development.  
2. The Jupyter Notebook is present in the following Kaggle [link](https://www.kaggle.com/code/kausthubkannan/character-generation-stylegan).  
3. The dataset which is trained upon is `Pixelated Treasures: 10K CryptoPunks` 
which can be found via this [link](https://www.kaggle.com/datasets/chwasiq0569/cryptopunks-pixel-art-dataset/).

## API Docs
Currently, the APIs support the generation of images and future expansion is the integration of LLMs for story generations.

### Endpoints

**Image Generation:** `/generated/{number_of_images}`  
The endpoint requires number_of_images to be entered. The response would be a single image which is a collage of a given 
number of images as shown in the image.  
  
Example:  

**JavaScript**
```javascript
import axios from "axios"

try{
  const number_of_images = 16
  const url = "http://127.0.0.1.8080/generated/{number_of_images}"
  const generated_image = axios.get()
}catch(err){
  console.log(err)
}
```

**Python**  
```python
import requests

try:
    number_of_images = 16
    url = f"http://127.0.0.1.8080/generated/{number_of_images}"
    response = requests.get(url)
    generated_image = response.content
except for Exception as err:
    print(err)

```
### Note
The model, which is based on **StyleGAN**, requires high memory in terms of CPU and CUDA. Hence, please `restrain from generating a number of images more than the range of (8, 48)`




