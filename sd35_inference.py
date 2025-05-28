import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist

from torchvision import transforms
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel


from torch.nn.parallel import DistributedDataParallel as DDP

from pipelines.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline

device = torch.device('cuda')


def get_args():
      parser = argparse.ArgumentParser()
      parser.add_argument("--prompt", default="Photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee.", type=str)
      
      parser.add_argument("--height", default=1024, type=int)
      parser.add_argument("--width", default=1024, type=int)

      parser.add_argument("--guidance-scale", default=4.5, type=float)
      parser.add_argument("--inference_step", default=50, type=int)

      parser.add_argument("--merge-ratio", default=0., type=float)
      parser.add_argument("--neural-left", default=0.3, type=float)
      parser.add_argument("--neural-right", default=0.7, type=float)
      parser.add_argument("--cfg-left", default=9.0, type=float)
      parser.add_argument("--cfg-right", default=-1, type=float)
      parser.add_argument("--inv_steps", default=28, type=int)

      parser.add_argument("--offset", default=2, type=int)

      args =  parser.parse_args()
      return args


if __name__ == '__main__':

      dtype = torch.float16
      args = get_args()
      print("-"*100)
      print(args)
      print("-"*100)

      prompt = args.prompt

      model_id = "stabilityai/stable-diffusion-3.5-large"

      nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                  )
      
      model_nf4 = SD3Transformer2DModel.from_pretrained(
                  model_id,
                  subfolder="transformer",
                  quantization_config=nf4_config,
                  torch_dtype=torch.bfloat16
            )

      pipe = StableDiffusion3Pipeline.from_pretrained(
                  model_id, 
                  transformer=model_nf4,
                  torch_dtype=torch.bfloat16
                  )
      pipe.enable_model_cpu_offload()

      original_img = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.inference_step).images[0]
      
      original_img.save(prompt[:15] + '_sd35' + '_standard.jpg')

      optim_img = pipe.forward_ours(
            prompt=prompt, 
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale, 
            num_inference_steps=args.inference_step,           
            cfg_gap_list=[args.cfg_left, args.cfg_right],
            offset=args.offset,
            merge_ratio=args.merge_ratio,
            neural_ratio=[args.neural_left, args.neural_right],
            steps=args.inv_steps,
      ).images[0]


      optim_img.save(prompt[:15] + '_sd35' + '_optim.jpg')
      
      new_width = original_img.width + optim_img.width
      new_image = Image.new("RGB", (new_width, original_img.height))
      new_image.paste(original_img, (0, 0))
      new_image.paste(optim_img, (original_img.width, 0))
      # 保存拼接后的图片
      new_image.save(prompt[:15] + '_sd35' + '_show.jpg')
