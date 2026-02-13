import argparse
import os
# import random
# import socket
import yaml
import torch
# import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
# import torchvision
# import models
import datasets
# import utils
from models import DenoisingDiffusion
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
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


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    train_dl = DATASET.get_loader()
    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    wandb_logger = WandbLogger(
        project = 'unet_i2i',
        name = "wd_ad5c3s",
        save_dir = 'logs',
        log_model = True,
        notes = f"Training with art_dataset toy combinatorics version. Mask only inside model",
    )
    ckpt = ModelCheckpoint(
        monitor = 'train_loss',
        save_top_k = 1,
        mode = 'min',
        filename=f"wd_ad6005c3s"
    )

    early_stop = EarlyStopping(
        monitor = config.training.monitor,
        patience = config.training.patience,
        mode = 'min',
        verbose = True
    )

    trainer = pl.Trainer(max_epochs = config.training.n_epochs,
                            devices = config.training.num_devices,
                            precision = config.training.precision,
                            accumulate_grad_batches = config.training.accumulate_grad_batches,
                            logger = wandb_logger,
                            callbacks = [early_stop, ckpt],
                            log_every_n_steps = 10,
                            accelerator = 'gpu'
                            # reload_dataloaders_every_n_epochs = 1,
                            )
    trainer.fit(diffusion, train_dataloaders = train_dl)



if __name__ == "__main__":
    main()
