import os
import time
# import glob
import numpy as np
# import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
import pytorch_lightning as pl

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


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

    #IMPORTANT if training on GPU (modded this part)
    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to('cuda')

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        # if isinstance(module, nn.DataParallel):
        #     inner_module = module.module
        #     module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
        #     module_copy.load_state_dict(inner_module.state_dict())
        #     module_copy = nn.DataParallel(module_copy)
        # else:
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(x0.dtype)
    CH_SPLIT = 3
    #modded, considering mask explicitly inside the model only rn
    x = x0[:, CH_SPLIT:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    # print(x0.dtype, x.dtype, t.dtype)
    output = model(torch.cat([x0[:, :CH_SPLIT, :, :], x], dim=1), t)
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, args, config):
        super(DenoisingDiffusion, self).__init__()
        self.args = args
        self.config = config

        # if config.training.precision== 'bf16-mixed':
        #     self.dtype = torch.bfloat16

        self.model = DiffusionUNet(config)#.to(self.dtype)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        # self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.register_buffer("betas", torch.from_numpy(betas).float())
        # betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = len(betas)

    def configure_optimizers(self):
        optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        #scheduler to add
        return optimizer
    
    def forward(self, x, t, e):
        return noise_estimation_loss(self.model, x, t, e, self.betas)
    
    def training_step(self, batch, batch_idx):
        x, _, __ = batch
        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
        x = x.to(device = self.device, dtype = self.dtype)
        x = data_transform(x)#7 channels
        e = torch.randn_like(x[:, 4:, :, :], dtype=self.dtype, device=self.device)#modded
        n =x.size(0)

         # antithetic sampling
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device = self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        
        loss = self(x, t, e)

        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        print('Updating ema')
        self.ema_helper.update(self.model)