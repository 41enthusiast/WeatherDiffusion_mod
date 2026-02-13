import torch
import numpy as np
import argparse
import os
import yaml
from yaml import CLoader as Loader
from models.unet import DiffusionUNet
from torchvision.transforms.functional import crop
from torch import nn
from utils import data_transform, inverse_data_transform, compute_alpha
from PIL import Image
import gradio as gr
import time
import gc
import matplotlib.pyplot as plt
import cv2

def load_image(file):
    if file is None:
        return None
    return file.name

def make_masked(img, mask):
    return img * (1 - mask[:, :, None] / 255.0) + mask[:, :, None] / 255.0 * 255.0


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
        # x0_preds = []
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            #denoising steps here
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            print('a_t', at, 't',t,'next_t', next_t)
            xt = xs[-1]#.to('cuda')
            et_output = torch.zeros_like(x_cond, device=x.device)
            
            manual_batching_size = 16#64
            
            # xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            # x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
            for p in range(0, len(corners), manual_batching_size):
                current_batch_corners = corners[p:p+manual_batching_size]
                # This only uses VRAM for <manual_batching_size/64> patches, not the whole image
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in current_batch_corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in current_batch_corners], dim=0)
                outputs = model(torch.cat([x_cond_patch, 
                                            xt_patch], dim=1), t)
                # outputs_cpu = outputs.cpu()
                for idx, (hi, wi) in enumerate(corners[p:p+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
                    if idx%50 -1 == 0:
                        current_grid = torch.div(et_output, x_grid_mask).to('cpu')
                        yield f"Step {i}: Filling Patch {p+idx+1}", inverse_data_transform(current_grid), gr.update() # flag as intermediate step patch filling
                #forced intermediate garbage collection
                # del xt_patch, x_cond_patch, outputs
                # torch.cuda.empty_cache()
                    
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = torch.clamp(x0_t, -1.0, 1.0)#clamp to prevent overflows - make to -1 to 1
            # x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            step_img = inverse_data_transform(xt_next.to('cpu'))
            yield f"Completed Step {i}/{len(seq)}", step_img, gr.update(value=None) #to display current denoising step number and intermediate output
            xs = [xt_next]
            print('c1', (c1).min(), (c1).max())
            print('xt_next part 2', (xt_next).min(), (xt_next).max())
            print('c2', (c2).min(), (c2).max())
            # break
            # xs.append(xt_next.to('cpu'))

            #forced garbage collection step to help memory issues
            # del et_output,et, xt, x0_t, t, next_t, at, at_next
            # torch.cuda.empty_cache()
            # gc.collect()

            # break # for sanity check
    xs = xs[-1].detach().to('cpu')#.int()
    print('Range of final output is ', xs.min(), xs.max())
    yield "ALL Completed", gr.update(value=None), xs #[-1], x0_preds

def tensor_to_pil(tensor):
    # tensor shape [1, 3, H, W]
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def diff_heatmap(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff = diff.mean(axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

def run_reverse_diffusion(masked_image, gt_img):
    print('Reverse diffusion process starts here')
    seed = SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    p_size = config.data.image_size
    start = time.time()
    x_cond = torch.from_numpy(masked_image).permute(2,0,1).float().to(device) / 255.0
    x_cond = x_cond.unsqueeze(0)
    x_rand = torch.rand(x_cond.size(), device = device)
    print('Making an overalapping patches grid', time.time()-start, 'seconds')#~7ms
    start = time.time()
    h_list, w_list = overlapping_grid_indices(x_cond, p_size)#, r)
    corners = [(i,j) for i in h_list for j in w_list]
    print(time.time()-start, 'seconds')
    start = time.time()
    skip = config.diffusion.num_diffusion_timesteps // SAMPLING_TIMESTEPS
    seq = range(0, config.diffusion.num_diffusion_timesteps, skip)
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    betas = torch.from_numpy(betas).float().to(device)
    x_grid_mask = torch.zeros_like(x_cond, device=device)+1e-6 #stability
    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1 
    print('Basic variables for denoising set up for an image', time.time()-start, 'seconds')#~50ms

    # 3. Call the generator and delegate yields to Gradio
    step = 0
    for status_text, step_img, final_img in generalized_steps_overlapping(
        x_rand, x_cond, seq, model, betas, x_grid_mask, corners=corners, p_size = p_size):
        start = time.time()
        
        # Convert torch tensors to PIL for Gradio display
        # grid_pil = tensor_to_pil(grid_img) if torch.is_tensor(grid_img) else grid_img
        step_pil = tensor_to_pil(step_img.detach()) if torch.is_tensor(step_img) else step_img
        final_pil = tensor_to_pil(final_img) if torch.is_tensor(final_img) else final_img
        if step==0:
            print('all images processing time ', time.time()-start, 'seconds')
            print('mask grid number of overlap is :', x_grid_mask.max())
        step += 1
        yield status_text+f"\n {step} patch groups done", tensor_to_pil(x_grid_mask/x_grid_mask.max()), step_pil, final_pil, gr.update(value=None)
    else:
        print(status_text, final_pil)
        heatmap_img = diff_heatmap(np.array(final_pil), gt_img)
        plt.imsave('final_restored.jpeg', final_pil)
        yield status_text, tensor_to_pil(x_grid_mask.detach()/x_grid_mask.max()), gr.update(value=None), final_pil, heatmap_img
    
    
CKPT = 'ckpts/WeatherDiff64.pth.tar'
SEED = 61
R = 16
SAMPLING_TIMESTEPS = 50#25
start = time.time()
with open(os.path.join("configs", "allweather.yml"), "r") as f:
    config = yaml.load(f, Loader = Loader)#not safe for arbitrary execution, but the loading time was painful
config = dict2namespace(config)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
torch.backends.cudnn.benchmark = True
print('Loaded configurations', time.time()-start, 'seconds')

print('Starting loading of model')
start = time.time()
checkpoint = torch.load(CKPT, map_location = device)
model = DiffusionUNet(config).to(device)
model.eval()
model_ckpt = unwrap_modelckpt(checkpoint['state_dict'])
model.load_state_dict(model_ckpt)
print('Loaded Model', time.time()-start, 'seconds')
