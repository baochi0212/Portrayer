# Portrayer
Long-generation Multimodal Large Language Model

## Demo
![Demo Screensholt](https://github.com/baochi0212/Portrayer/blob/master/demo_portrayer.giff.gif)

## Quickstart Code
Provide a simple, complete code snippet that demonstrates how to quickly get started with your project. This should include the minimal setup or usage example needed to experience the core functionality or a key feature of the project.
```
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import TextStreamer
from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "chitb/portrayer_6k_dev"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "./joker.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWrite a long and well-formated passage to advertise this movie, around 1000 words."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
```
## Citation

If you find our project useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{portrayer,
      title={Portrayer: Long-generation Multimodal Large Language Model}, 
      author={Chi Tran},
      publisher={github repo},
      year={2024}
}
```
## Acknowledgement 
[LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT)
[Longwriter](https://github.com/THUDM/LongWriter) 

