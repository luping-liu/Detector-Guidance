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
        default="pose",
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
        choices=["yolo", ]
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


def train_yolo(diffusion, yolo, dataloader, accelerator, tb_writer, args):
    # if args.task == 'pose':
    #     yolo.load_state_dict(torch.load(f'{save_dir}/model/11112140-pose-yolo/48000.ckpt', map_location='cpu'))
    # #     attn_control.load_state_dict(torch.load(f'{save_dir}/model/09221546-pose-atoken/15000.ckpt', map_location='cpu'))
    # #     # yolo.load_state_dict(torch.load('runs/model/09091715-pose-yolo/30000.ckpt', map_location='cpu'))
    # elif args.task == 'text':
    #     # attn_control.load_state_dict(torch.load(f'{save_dir}/model/09221746-text-atoken/15000.ckpt', map_location='cpu'))
    #     yolo.load_state_dict(torch.load(f'{save_dir}/model/11112140-text-yolo/48000.ckpt', map_location='cpu'))

    # attn_control.set_atoken(args.task)
    # if 'sd-xl-base' in args.sd_type:
    #     attn_control = attn_control_sgm
    # attn_control = attn_control.to(device, precision)
    diffusion = diffusion.to(device, precision)  # accelerator.prepare_model does not work well
    diffusion.eval()

    optimizer = torch.optim.AdamW(params=yolo.parameters(), lr=cfg.train.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: min(1., t / cfg.train.warmup_steps))

    yolo, optimizer, lr_scheduler, dataloader = accelerator.prepare(yolo, optimizer, lr_scheduler, dataloader)
    if args.resume:
        accelerator.load_state(f'{save_dir}/state/{args.resume}')
        with open(f'{save_dir}/state/{args.resume}/step.txt', 'r') as f:
            step = int(f.read())
        print(f'resume state from {args.resume} with step {step}')
    else:
        step = 0

    print("*******************")
    print('start training ...')
    import warnings
    warnings.filterwarnings('ignore')

    for epoch in range(50):
        dataset.shuffle_dataset_hymie()
        for i, batch in enumerate(dataloader):
            with torch.no_grad(), accelerator.autocast():
                batch['jpg'] = batch['jpg'].to(device, precision) * 2 - 1
                # batch['hint'] = batch['pose_hint'].to(device, precision)
                # batch['mask'] = batch['focus'].to(device, precision)  # [16, 512, 512] [0, 1]
                batch['seg'] = batch['seg'].to(device, precision)
                # import pdb; pdb.set_trace()
                attn_control.register_index(batch['txt_mask'])
                # attn_control.token_index = batch['txt_mask']
                _, loss_dict = diffusion.shared_step(batch)
                z_pred = loss_dict['val/x_start_pred'].clamp(-2.5, 2.5)  # reduce big number
                # z_noisy = loss_di'val/x_noisy'].clamp(-2.5, 2.5)
                timestep = loss_dict['val/timestep'].cpu().float().mean()
                cams = attn_control.extract_cams()
                attn_control.clear()
                # import pdb; pdb.set_trace()
                for i in range(cfg.train.batch_size):
                    cams[i] = cams[i].index_select(0, batch['shuffle_ids'][i])
                cams = rearrange(cams, 'b i j ... -> b (i j) ...')
                batch['cls'] = batch['order']
                # import pdb; pdb.set_trace()
                # yolo_input_dict = {'attn': torch.cat([z_pred, cams[:, 2:]], dim=1).detach()}
                yolo_input_dict = {'attn': cams.detach()}
                yolo_input_dict = to_yolo_input(args.task, yolo_input_dict, batch=batch, device=device, precision=precision)

            with accelerator.autocast():
                loss, loss_list = yolo(yolo_input_dict)

            # continue
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(yolo.parameters(), 2.0)
            optimizer.step()
            # attn_control.clear()
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            optimizer.zero_grad()
            step += 1

            # if args.task == 'pose':
            #     loss_dict = {'loss': loss, 'bbox': loss_list[0], 'pose': loss_list[1], 'obj': loss_list[2],
            #                  'cls': loss_list[3], 'dfl': loss_list[4], 'de': loss_list[5], 'ge': loss_list[6],
            #                  't': timestep}
            # elif args.task == 'text' or :
            loss_dict = {'loss': loss, 'bbox': loss_list[0], 'mask': loss_list[1], 'cls': loss_list[2],
                         'dfl': loss_list[3], 't': timestep}

            if global_rank == 0:
                if step % 20 == 0:
                    print(f'step: {step}, lr: {lr_scheduler.get_last_lr()[0]:.7f}', end='')
                    tb_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], step)
                    for k, v in loss_dict.items():
                        print(f', {k}: {v.item():.5f}', end='')
                        tb_writer.add_scalar(f'train/{k}', v.item(), step)
                    print('')
                if step % 3000 == 0 and not args.debug:
                    unwrapped_model = accelerator.unwrap_model(yolo)
                    torch.save(unwrapped_model.state_dict(), f'{save_dir}/model/{exp_time}-{args.task}-{args.stage}/{step}.ckpt')
                    # accelerator.save_state(output_dir=f'{save_dir}/state/{exp_time}-{args.task}-{args.stage}')
                    # with open(f'{save_dir}/state/{exp_time}-{args.task}-{args.stage}/step.txt', 'w') as f:
                    #     f.write(str(step))

            # import pdb; pdb.set_trace()
            # samples_z = diffusion.sample(val_c, val_uc, 2, (4, 64, 64))
            # samples_x = diffusion.decode_first_stage(samples_z)
            # print('pass')

            if step == cfg.train.total_step:
                accelerator.wait_for_everyone()
                sys.exit()


def eval_yolo():
    pass


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
    dataset = HunyuanDatasetStream(img_size=cfg.train.image_size, data_type=data_type, arrow_files=arrow_files,
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

    # log
    if args.resume:
        exp_time, task_, stage_ = args.resume.split('-')
        assert task_ == args.task and stage_ == args.stage
    else:
        exp_time = datetime.datetime.now().strftime('%m%d%H%M')
        if global_rank == 0 and not args.debug:
            os.makedirs(f'{save_dir}/model/{exp_time}-{args.task}-{args.stage}', exist_ok=True)
            os.makedirs(f'{save_dir}/state/{exp_time}-{args.task}-{args.stage}', exist_ok=True)

    if global_rank == 0:
        if not args.debug:
            tb_writer = SummaryWriter(f'{save_dir}/board/{exp_time}-{args.task}-{args.stage}')
        else:
            tb_writer = SummaryWriter(f'/tmp/tmp_{exp_time}')
    else:
        tb_writer = None

    # train
    if args.stage == 'yolo':
        train_yolo(diffusion, yolo, dataloader, accelerator, tb_writer, args)
    else:
        pass




