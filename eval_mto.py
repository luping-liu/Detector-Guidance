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

from dataset import HunyuanDatasetStream
from utils import collate_fn_pose, collate_fn_text, collate_fn_mto, to_yolo_input
from stablediffusion import Diffusion, ControlNet, attn_control
from ultralytics import YOLO
# from stablediffusion.annotator.ppocr import MyRecModel
from stablediffusion.cldm.recognizer import create_predictor
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
        "--task",
        type=str,
        default="mto",
        choices=['mto']  # segment <-> visual text, pose <-> human image
    )
    parser.add_argument(
        "--data",
        type=str,
        default="good_hand",
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
        choices=["atoken", "yolo", "controlnet"]
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


def eval_yolo(diffusion, yolo, dataloader, accelerator, args):
    yolo.load_state_dict(torch.load(f'{save_dir}/model/01221903-mto-yolo/72000.ckpt', map_location='cpu'))

    yolo = AutoBackend(yolo, data=f'yolov8l-{args.task}.yaml', verbose=False,
                       device=device, fp16=(precision == torch.float16))
    yolo.postprocess = yolo_process
    yolo.eval()

    diffusion = diffusion.to(device, precision)  # accelerator.prepare_model does not work well
    diffusion.eval()

    for i, batch in enumerate(dataloader):
        if i < 15:
            continue
        with torch.no_grad(), accelerator.autocast():
            batch['jpg'] = batch['jpg'].to(device, precision) * 2 - 1
            # batch['hint'] = batch['pose_hint'].to(device, precision)
            # batch['mask'] = batch['focus'].to(device, precision)  # [16, 512, 512] [0, 1]
            batch['seg'] = batch['seg'].to(device, precision)
            # import pdb; pdb.set_trace()
            attn_control.register_index(batch['txt_mask'].to(device))
            # attn_control.token_index = batch['txt_mask']
            _, loss_dict = diffusion.shared_step(batch)
            z_pred = loss_dict['val/x_start_pred'].clamp(-2.5, 2.5)  # reduce big number
            timestep = loss_dict['val/timestep'].cpu().float().mean()
            cams = attn_control.extract_cams()
            attn_control.clear()
            # import pdb; pdb.set_trace()
            batch['shuffle_ids'] = batch['shuffle_ids'].to(device)
            for k in range(cfg.train.batch_size):
                cams[k] = cams[k].index_select(0, batch['shuffle_ids'][k])
            cams = rearrange(cams, 'b i j ... -> b (i j) ...')
            batch['cls'] = batch['order']
            # import pdb; pdb.set_trace()
            # yolo_input_dict = {'attn': torch.cat([z_pred, cams[:, 2:]], dim=1).detach()}
            yolo_input_dict = {'attn': cams.detach()}
            yolo_input_dict = to_yolo_input(args.task, yolo_input_dict, batch=batch, device=device, precision=precision)

            output = yolo(yolo_input_dict['attn'])
            results = yolo.postprocess(output, yolo_input_dict['attn'], torch.zeros([512, 512]))
            for k in range(6):
                plt.subplot(2, 3, k+1)
                label = [str(m) for m in results[k].boxes.cls.int().cpu().tolist()]
                vis = draw_bounding_boxes(((batch['jpg'][k].float().cpu() * 0.5 + 0.5) * 255).to(torch.uint8), results[k].boxes.xyxy.cpu(),
                                          label, width=6, font='./simfang.ttf', font_size=48)
                plt.imshow(vis.permute(1, 2, 0).numpy())
                plt.axis('off')
                # bbox = results[k].boxes.xyxy.cpu()
                # plt.plot(bbox[:, [0, 2, 2, 0, 0]].T, bbox[:, [1, 1, 3, 3, 1]].T, '-')
            plt.savefig('tmp/step.jpg')
            plt.close()

            import pdb; pdb.set_trace()
            print('pass')


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(f'config/{args.stage}.yaml')
    precision = torch.float16 if cfg.train.mixed_precision == 'fp16' else torch.float32
    save_dir = '/apdcephfs_cq5/share_300167803/lupingliu/Workspace/CycleNet/runs-icml'

    # data
    data_type = args.data.replace(' ', '').split(',')
    if args.index_file is not None:
        with open(args.index_file, 'r') as f:
            res_dict = json.load(f)
        arrow_files = res_dict['arrow_files']
        indexs = res_dict['indexs']
    else:
        arrow_files, indexs = None, None
    dataset = HunyuanDatasetStream(img_size=cfg.train.image_size, split='eval', data_type=data_type, arrow_files=arrow_files,
                                   indexs=indexs)
    if args.task == 'mto':
        collate_fn_ = collate_fn_mto
    else:
        collate_fn_ = collate_fn_pose
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=6,
                                             drop_last=True, pin_memory=True, collate_fn=collate_fn_, prefetch_factor=3)

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
        eval_yolo(diffusion, yolo, dataloader, accelerator, args)
    else:
        pass
