import os
import random
import sys
import json
import time
import datetime
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import set_seed
from deepspeed import init_inference
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from diffusers.utils.torch_utils import randn_tensor

# from dataset import HunyuanDatasetStream
from utils import collate_fn_mto
from utils import prompt_parser
from stablediffusion import Diffusion, ControlNet, attn_control
from stablediffusion import DDIMSampler, DPMSolverSampler
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend


rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
if world_size > 1:
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_type",
        type=str,
        default="sd-2.1",
        choices=["sd-2.1", 'sd-2.1-control', "sd-xl-base", 'sd-xl-base-control']
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mto",
        choices=['mto']  # segment <-> visual text, pose <-> human image
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="yolo",
        choices=["yolo",]
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    opt = parser.parse_args()
    return opt


def sample_mto(diffusion, yolo, accelerator, args):
    yolo.load_state_dict(torch.load(f'store/detector_guidance_yolo_sd2.ckpt', map_location='cpu'))

    yolo = AutoBackend(yolo, data=f'yolov8l-{args.task}.yaml', verbose=False,
                       device=device, fp16=(precision == torch.float16))
    yolo.postprocess = yolo_process
    yolo.eval()

    diffusion = diffusion.to(device, precision)  # accelerator.prepare_model does not work well
    diffusion.eval()

    sampler = DDIMSampler(diffusion, device=device)
    sampler.yolo = yolo

    batch_size = 1
    input_shape = [512, 512]
    # caption_list = (["A white cat and a brown dog"] * 5 + ["a rusty robot and an elegant doll"] * 5 +
    #            ["A striped tiger and a spotted leopard"] * 5)
    # with open('/apdcephfs/share_1367250/lupingliu/Workspace/Rebuttal/prompt_list.txt', 'r') as f:
    #     prompts = f.readlines()
    # prompts = [p.strip() for p in prompts][rank::world_size]
    if args.prompt is not None:
        prompts = [args.prompt.strip()]
    else:
        prompts = ['a regal lion and a sly fox',
                   'a wise owl and a nimble squirrel',
                   'a striped tiger and a spotted leopard'][rank::world_size]

    with torch.no_grad(), accelerator.autocast():
        for i, prompt in enumerate(tqdm(prompts)):
            # caption = prompt.replace('and', 'plays with')
            caption = prompt

            tokens = torch.zeros(77)
            tokens_sub = torch.zeros(77)
            ids = prompt_parser(caption)
            for m, n in enumerate(ids[0]):
                tokens[n] = m + 1
            for m, n in enumerate(ids[1]):
                tokens_sub[n] = m + 1

            for j in range(100):

                uc = diffusion.get_learned_conditioning([""] * batch_size)
                c = diffusion.get_learned_conditioning([caption] * batch_size)

                shape = [4, input_shape[0] // 8, input_shape[0] // 8]
                # x_T = torch.randn(batch_size, *shape).to(device)
                generator = torch.Generator("cuda").manual_seed(666 + j)
                x_T = randn_tensor([1, 4, 64, 64], generator=generator, device=device, dtype=precision)
                # x_T = torch.load('temp/x_T.pt')
                # if cooo == 3:
                #     torch.save(x_T, 'temp/x_T.pt')
                #     print('checkpoint')

                attn_control.register_index(tokens[None, ...].to(device), tokens_sub[None, ...].to(device))

                print(j, caption)
                samples1, intermediates = sampler.sample(S=50, conditioning=c, batch_size=batch_size,
                                                         shape=shape, x_T=x_T, verbose=False,
                                                         unconditional_guidance_scale=7.0,
                                                         unconditional_conditioning=uc, eta=0)

                x_samples = diffusion.decode_first_stage(samples1)  # [-1, 1]
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)[0]

                # plt.imsave(f"/apdcephfs/share_1367250/lupingliu/Workspace/Rebuttal/image/detector-guidance-play/{rank + i * world_size}_{666 + j}.jpg",
                #            x_samples.permute(1, 2, 0).cpu().numpy())
                plt.imsave('step.jpg', x_samples.permute(1, 2, 0).cpu().numpy())
                # import pdb; pdb.set_trace()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(f'config/{args.stage}.yaml')
    precision = torch.float16 if cfg.train.mixed_precision == 'fp16' else torch.float32

    # data
    data_type = args.data.replace(' ', '').split(',')
    if args.index_file is not None:
        with open(args.index_file, 'r') as f:
            res_dict = json.load(f)
        arrow_files = res_dict['arrow_files']
        indexs = res_dict['indexs']
    else:
        arrow_files, indexs = None, None

    # model
    if args.sd_type == 'sd-2.1-control':
        diffusion = ControlNet(args.sd_type, device=device)
    else:
        diffusion = Diffusion(args.sd_type, device=device, verbose=True)

    yolo_task = 'segment' if args.task == 'mto' else args.task
    yolo = YOLO(f'yolov8l-{args.task}.yaml', yolo_task)
    yolo.prepare(data=f'coco8-{args.task}.yaml', batch=8 * world_size, epochs=100, imgsz=cfg.train.image_size, device=device)
    yolo_process = yolo.predictor.postprocess
    yolo = yolo.model

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)

    # train
    if args.stage == 'yolo':
        sample_mto(diffusion, yolo, accelerator, args)
    else:
        pass
