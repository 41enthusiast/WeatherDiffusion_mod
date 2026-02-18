import argparse
import os
import yaml
import torch
import numpy as np
from models.unet import DiffusionUNet
from torchvision.transforms.functional import crop
from torch import nn
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from tqdm import tqdm

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='ckpts/WeatherDiff64.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def overlapping_grid_indices(x_cond, output_size, r=None):
    _, _, h, w = x_cond.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    return h_list, w_list

def data_transform(X):#used at generalized_steps_overlapping
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps_overlapping(x, x_cond, masked_tensor, mask_tensor, seq, model, b, x_grid_mask, eta=0., corners=None, p_size=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        gif_frames = []
        if mask_tensor is not None:
            mask_tensor = mask_tensor.to(x.device)
        masked_tensor = masked_tensor.to(x.device)

        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq), desc="Denoising Steps"):
            #denoising steps here
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et_output = torch.zeros_like(x, device=x.device)
            
            manual_batching_size = 64
            patch_pbar = tqdm(range(0, len(corners), manual_batching_size), 
                          leave=False, 
                          desc=f"Processing Patches (t={i})")
            for p in patch_pbar:
                current_batch_corners = corners[p:p+manual_batching_size]
                
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in current_batch_corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in current_batch_corners], dim=0).to(x.device)
                
                outputs = model(torch.cat([x_cond_patch, 
                                            xt_patch], dim=1), t)
                for idx, (hi, wi) in enumerate(corners[p:p+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]  
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            # xs.append(xt_next.to('cpu'))
            # xs = [xt_next*(1-mask_tensor) + (mask_tensor)*masked_tensor]#to reuse good parts of the image
            xs = [xt_next]#original no reuse
            assert xt_next.max() <=20, "Exploding values, something is wrong at"+i
            # --- GIF LOGIC ---
            # Convert the current xt to a viewable image and store it
            current_img = inverse_data_transform(xt).squeeze().cpu()
            # Convert to PIL Image: (C, H, W) -> (H, W, C)
            gif_frames.append(to_pil_image(current_img))
    return xs, gif_frames#, x0_preds

def unwrap_modelckpt(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove 'module.'
        # name = k[6:] if k.startswith('model.') else k # remove 'model.' for pytorch lightning port
        new_state_dict[name] = v
    return new_state_dict

def main(masked_img, masked_tensor, mask_tensor = None):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    masked_img.unsqueeze_(0)#dim will be 4 here - evaluation phase
    print(CKPT)
    checkpoint = torch.load(CKPT)#, map_location=device)
    model = DiffusionUNet(config)
    model.to(device)

    model_ckpt = unwrap_modelckpt(checkpoint['state_dict'])
    model.load_state_dict(model_ckpt, strict = False)
    model.eval()

    # if 'ema_helper' in checkpoint:
    #     print('Found EMA')
    #     ema_helper = EMAHelper()
    #     ema_helper.load_state_dict(checkpoint['ema_helper'])
    #     ema_helper.ema(model) # This copies the smooth weights into your model

    r = args.grid_r
    p_size = config.data.patch_size
    x_rand = torch.randn([1,3,]+list(masked_img.shape[2:]), device = config.device)
    # print(x_rand.shape, masked_img.shape)
    h_list, w_list = overlapping_grid_indices(masked_img, p_size, r)
    corners = [(i,j) for i in h_list for j in w_list]
    print('Number of patches', len(corners))
    skip = config.diffusion.num_diffusion_timesteps // args.sampling_timesteps
    seq = range(0, config.diffusion.num_diffusion_timesteps, skip)
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    betas = torch.from_numpy(betas).float().to(device)
    x_grid_mask = torch.zeros_like(x_rand, device=device)
    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1 
    xs, gif_frames = generalized_steps_overlapping(x_rand, masked_img, masked_tensor, mask_tensor, seq, model, betas, x_grid_mask, eta=0.,
                                                            corners=corners, p_size=p_size)
    xs = xs[-1]
    gif_frames[0].save(
        'reverse_diffusion.gif',
        save_all=True,
        append_images=gif_frames[1:],
        duration=100, 
        loop=0
    )
    print('Xs range', xs.min(), xs.max())
    return inverse_data_transform(xs)

if __name__ == '__main__':
    args, config = parse_args_and_config()
    CKPT = 'ckpts/WeatherDiff64.pth.tar'
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device
    img_file = '../raindrop_data/test_a/data/0_rain.png'
    # mask_file = '../raindrop_data/test_a/mask/2a5a6ce95109caba13b6c840ed22638f.png'
    masked_img = Image.open(img_file).convert('RGB')
    # mask_img = Image.open(mask_file).convert('L')
    masked_tensor = to_tensor(masked_img)
    # mask_tensor = to_tensor(mask_img)
    # torch.cat([masked_tensor,mask_tensor], dim = 0), , mask_tensor
    result = main(masked_tensor, masked_tensor)
    print(result.shape, result.min(), result.max())
    result = to_pil_image(result.squeeze().cpu())
    os.makedirs(f"outputs/{CKPT.split('/')[-1].split('.')[0]}", exist_ok = True)
    plt.imsave(f"outputs/{CKPT.split('/')[-1].split('.')[0]}/{img_file.split('/')[-1]}", result)
