import sys
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from open_clip.tokenizer import SimpleTokenizer
sys.path.append('./stablediffusion')
from ldm.util import instantiate_from_config
from cldm.model import create_model, load_state_dict
from ldm.modules.attention import attn_control

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


diffusion_dict = {
    'sd-2.1': ['stablediffusion/configs/stable-diffusion/v2-inference.yaml', 'store/v2-1_512-ema-pruned.safetensors'],
    'sd-2.1-control': ['stablediffusion/configs/stable-diffusion/control-sd-2-1.yaml', 'store/v2-1_512-ema-pruned-controlnet.ckpt'],
    'sd-xl-base': ['stablediffusion/configs/controlnet-xs/sdxl-base.yaml', 'store/sd_xl_base_1.0.safetensors'],
    'sd-xl-base-control': ['stablediffusion/configs/controlnet-xs/sdxl-full-pose-20.yaml', 'store/sd_xl_base_1.0.safetensors'],
}


def Diffusion(mtype, device=torch.device("cpu"), verbose=False):
    config, ckpt = diffusion_dict[mtype]
    config = OmegaConf.load(config)

    if ckpt.endswith("ckpt"):
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_file(ckpt)
    else:
        sd = {}
    # sd = {k.replace("cond_stage_model.", "conditioner.embedders.0."): v for k, v in sd.items()}

    model = instantiate_from_config(config.model)

    if mtype == 'sd-xl-base-control':
        m, u = model.load_state_dict(sd, strict=False)
        sd_new_ = model.state_dict()
        sd_new = {}
        for k in sd_new_.keys():
            if 'model.control_model.' in k and 'flag' not in k:
                ori_k = k.replace('model.control_model.', 'model.diffusion_model.')
                # import pdb; pdb.set_trace()
                if ori_k in sd:
                    # if sd_new[k].shape != sd[ori_k].shape:
                    #     import pdb; pdb.set_trace()
                    #     print(sd_new[k].shape, sd[ori_k].shape)
                    sd_new[k] = sd[ori_k].clone()
        # import pdb; pdb.set_trace()
        _, u = model.load_state_dict(sd_new, strict=False)
    else:
        m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # influence the train of controlnet?
    # if 'sd-xl-base' in mtype:
    #     vae = load_file('store/diffusion/sdxl_vae_fix.safetensors')
    #     m, u = model.first_stage_model.load_state_dict(vae, strict=False)
    #
    #     if len(m) > 0 and verbose:
    #         print("missing keys:")
    #         print(m)
    #     if len(u) > 0 and verbose:
    #         print("unexpected keys:")
    #         print(u)

    # if device == torch.device("cpu"):
    #     model.cpu()
    #     model.cond_stage_model.device = "cpu"
    # else:
    #     model.to(device)

    return model


def ControlNet(mtype, device=torch.device("cuda"), verbose=True):
    config, ckpt = diffusion_dict[mtype]
    # Configs
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config).cpu()
    sd = load_state_dict(ckpt, location='cpu')
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # model.to(device)
    # model.eval()
    return model
