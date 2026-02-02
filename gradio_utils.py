import torch
import numpy as np
import argparse
import os
import yaml
from models.unet import DiffusionUNet
from torchvision.transforms.functional import crop
from torch import nn
from utils import data_transform, inverse_data_transform, compute_alpha
from PIL import Image
import gradio as gr

def load_image(file):
    if file is None:
        return None
    return file.name

def make_masked(img, mask):
    return img * (1 - mask[:, :, None] / 255.0) + mask[:, :, None] / 255.0 * 255.0

CKPT = 'ckpts/WeatherDiff64.pth.tar'
SEED = 61
R = 16
SAMPLING_TIMESTEPS = 25

def unwrap_modelckpt(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove 'module.'
        new_state_dict[name] = v
    return new_state_dict

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def overlapping_grid_indices(x_cond, output_size):#, r=None):
    _, _, h, w = x_cond.shape
    r = R #if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    return h_list, w_list

def generalized_steps_overlapping(x, x_cond, seq, model, b, x_grid_mask, eta=0., corners=None, p_size=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            #denoising steps here
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            manual_batching_size = 64
            xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
            for i in range(0, len(corners), manual_batching_size):
                outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                            xt_patch[i:i+manual_batching_size]], dim=1), t)
                for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
                    current_grid = torch.div(et_output, x_grid_mask)
                    yield f"Step {i}: Filling Patches...", inverse_data_transform(current_grid), gr.update(), gr.update() # flag as intermediate step patch filling
                    
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            step_img = inverse_data_transform(xt_next.to('cpu'))
            yield f"Completed Step {i}", gr.update(value=None), step_img, gr.update() #to display current denoising step number and intermediate output
            xs.append(xt_next.to('cpu'))
    return xs#, x0_preds

def tensor_to_pil(tensor):
    # tensor shape [1, 3, H, W]
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def run_reverse_diffusion(masked_image, model):
    seed = SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = 'cuda'
    torch.backends.cudnn.benchmark = True

    with open(os.path.join("configs", "allweather.yml"), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    p_size = config.data.image_size

    checkpoint = torch.load(CKPT)
    model = DiffusionUNet(config)
    model.to(device)
    model.eval()
    model_ckpt = unwrap_modelckpt(checkpoint['state_dict'])
    model.load_state_dict(model_ckpt)

    x_cond = torch.from_numpy(masked_image).permute(2,0,1).float().to(device) / 255.0
    x_cond = x_cond.unsqueeze(0)
    x_rand = torch.rand(x_cond.size(), device = device)
    h_list, w_list = overlapping_grid_indices(x_cond, p_size)#, r)
    corners = [(i,j) for i in h_list for j in w_list]
    skip = config.diffusion.num_diffusion_timesteps // SAMPLING_TIMESTEPS
    seq = range(0, config.diffusion.num_diffusion_timesteps, skip)
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    betas = torch.from_numpy(betas).float().to(device)
    x_grid_mask = torch.zeros_like(x_cond, device=device)
    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1 

    # 3. Call the generator and delegate yields to Gradio
    for status_text, grid_img, step_img, final_img in generalized_steps_overlapping(
        x_rand, x_cond, seq, model, betas, x_grid_mask, corners=corners, p_size = p_size):
        
        # Convert torch tensors to PIL for Gradio display
        grid_pil = tensor_to_pil(grid_img) if torch.is_tensor(grid_img) else grid_img
        step_pil = tensor_to_pil(step_img) if torch.is_tensor(step_img) else step_img
        final_pil = tensor_to_pil(final_img) if torch.is_tensor(final_img) else final_img
        
        yield x_grid_mask, status_text, grid_pil, step_pil, final_pil
    
    
    