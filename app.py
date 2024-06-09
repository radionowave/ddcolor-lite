import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import subprocess
import shutil
import os
import requests

class ImageColorizationPipeline(object):
    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_name = 'convnext-t' if model_size == 'tiny' else 'convnext-l'
        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        self.height, self.width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()

        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)    

        return output_img

def download_model(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(path, 'wb') as file:
            for data in response.iter_content(block_size):
                size = file.write(data)
                progress_bar.update(size)
        progress_bar.close()
    else:
        print("Model already downloaded.")

def generate(image):
    image_in = cv2.imread(image)
    image_out = colorizer.process(image_in)
    cv2.imwrite('out.jpg', image_out)
    image_in_pil = Image.fromarray(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
    image_out_pil = Image.fromarray(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB))
    return image_in_pil, image_out_pil

model_url = "https://huggingface.co/camenduru/cv_ddcolor_image-colorization/resolve/main/pytorch_model.pt"
model_path = "models/pytorch_model.pt"
download_model(model_url, model_path)

colorizer = ImageColorizationPipeline(model_path=model_path, input_size=512)

inputs = [
    gr.inputs.Image(type="filepath", label="Upload Image"),
]

outputs = [
    gr.outputs.Image(type="pil", label="Input Image"),
    gr.outputs.Image(type="pil", label="Colorized Image")
]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Upload Image")
            button = gr.Button("Colorize")
        output_image = gr.Image(label="Colorized Image")
    button.click(fn=generate, inputs=[image], outputs=[output_image])

demo.launch(share=True)