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


from torch.nn.parallel import DistributedDataParallel as DDP

from pipelines.pipeline_flux import FluxPipeline

device = torch.device('cuda')


def get_args():
      parser = argparse.ArgumentParser()
      parser.add_argument("--prompt", default="Photorealistic closeup video of two pirate ships battling each other as they sail inside a cup of coffee.", type=str)
      
      parser.add_argument("--height", default=1024, type=int)
      parser.add_argument("--width", default=1024, type=int)

      parser.add_argument("--guidance-scale", default=4.5, type=float)
      parser.add_argument("--inference_step", default=28, type=int)
      parser.add_argument("--model", default='lite', type=str, choices=['lite', 'dev'])

      parser.add_argument("--merge-ratio", default=0., type=float)
      parser.add_argument("--neural-left", default=0.3, type=float)
      parser.add_argument("--neural-right", default=0.7, type=float)
      parser.add_argument("--cfg-left", default=9.0, type=float)
      parser.add_argument("--cfg-right", default=-1, type=float)
      parser.add_argument("--inv_steps", default=28, type=int)

      parser.add_argument("--offset", default=2, type=int)

      parser.add_argument("--accelerate", default=False, action='store_true')

      args =  parser.parse_args()
      return args


if __name__ == '__main__':

      dtype = torch.float16
      args = get_args()
      print("-"*100)
      print(args)
      print("-"*100)


      prompt = args.prompt

      if args.model == 'lite':
            model_id = "Freepik/flux.1-lite-8B-alpha"
      elif args.model == 'dev':
            model_id = "black-forest-labs/FLUX.1-dev"

      if not args.accelerate:  
            # load pipe
            pipe = FluxPipeline.from_pretrained(
                  model_id,
                  torch_dtype=torch.bfloat16,
            )
      else:
            from nunchaku import NunchakuFluxTransformer2dModel
            transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")

            # load pipe
            pipe = FluxPipeline.from_pretrained(
                  model_id,
                  transformer=transformer,
                  torch_dtype=torch.bfloat16,
            )

      pipe.enable_model_cpu_offload()  # Less VRAM or something


      original_img = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.inference_step).images[0]
      
      original_img.save(prompt[:15] + f"_{args.model}" + '_standard.jpg')

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


      optim_img.save(prompt[:15] + f"_{args.model}" + '_optim.jpg')
      
      new_width = original_img.width + optim_img.width
      new_image = Image.new("RGB", (new_width, original_img.height))
      new_image.paste(original_img, (0, 0))
      new_image.paste(optim_img, (original_img.width, 0))
      # 保存拼接后的图片
      new_image.save(prompt[:15] + f"_{args.model}" + '_show.jpg')
