import os
import argparse
import torch
from tqdm import tqdm

from utils import prompt_parser


rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
if world_size > 1:
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="sd2",
        choices=["sd2", 'ae', "db", "dg"]
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_args()
    model_path = './store/huggingface/stable-diffusion-2-1-base'

    if args.prompt is not None:
        prompts = [args.prompt.strip()]
    else:
        with open('prompt_list.txt', 'r') as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts][rank::world_size]

    if args.model == 'ae':
        from diffusers import StableDiffusionAttendAndExcitePipeline

        pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        full_name = 'attend-and-excite'
    elif args.model == 'db':
        from diffusersdev.pipeline_divide_and_bind_latest import StableDiffusionDivideAndBindPipeline

        pipe = StableDiffusionDivideAndBindPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        full_name = 'divide-and-bind'
    elif args.model == 'sd2':
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        full_name = 'stable-diffusion'
    elif args.model == 'dg':
        from ultralytics import YOLO
        from ultralytics.nn.autobackend import AutoBackend
        from diffusersdev.pipeline_detector_guidance import StableDiffusionDetectorGuidancePipeline

        pipe = StableDiffusionDetectorGuidancePipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        full_name = 'detector-guidance'

        yolo_task = 'segment'
        args.task = 'mto'
        yolo = YOLO(f'yolov8l-{args.task}.yaml', yolo_task)
        yolo.prepare(data=f'coco8-{args.task}.yaml', batch=8 * world_size, epochs=100, imgsz=512, device=device)
        yolo_process = yolo.predictor.postprocess
        yolo = yolo.model

        yolo.load_state_dict(torch.load(f'store/detector_guidance_yolo_sd2.ckpt', map_location='cpu'))

        yolo = AutoBackend(yolo, data=f'yolov8l-{args.task}.yaml', verbose=False, device=device, fp16=True)
        yolo.postprocess = yolo_process
        yolo.eval()

    # from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
    #
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    for i, prompt in enumerate(tqdm(prompts)):
        main_ids, attr_ids = prompt_parser(prompt)
        if args.model in ['ae', 'db']:
            attr_ids = [c[-len(m) - 1] for m, c in zip(main_ids, attr_ids)]
            main_ids = [m[-1] for m in main_ids]
            loss_mode = 'tv_bind'
        elif args.model == 'dg':
            tokens = torch.zeros(77)
            tokens_sub = torch.zeros(77)
            for m, n in enumerate(main_ids):
                tokens[n] = m + 1
            for m, n in enumerate(attr_ids):
                tokens_sub[n] = m + 1

        generator = torch.Generator("cuda").manual_seed(args.seed)

        add_kwargs = {}
        if args.model == 'ae':
            add_kwargs = {'token_indices': main_ids, 'max_iter_to_alter': 25}
        elif args.model == 'db':
            add_kwargs = {'token_indices': main_ids, 'color_indices': attr_ids, 'max_iter_to_alter': 25,
                          'loss_mode': loss_mode}
        elif args.model == 'dg':
            add_kwargs = {'token_indices': tokens[None, ...].to(device),
                          'attr_indices': tokens_sub[None, ...].to(device),
                          'detector': yolo}

        images = pipe(
            prompt=prompt,
            guidance_scale=7.5,
            generator=generator,
            num_inference_steps=50,
            **add_kwargs
        ).images

        image = images[0]
        image.save('step.jpg')

