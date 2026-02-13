import logging
import numpy as np
import torch
from PIL import Image
import torchvision
# from functools import lru_cache
# from functools import partial
# from itertools import repeat
# from multiprocessing import Pool
# from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# from torchvision.transforms import InterpolationMode
# from utils import show_image_mask_grid
import random
# import PIL
# import re
# import json
# from torchvision.transforms import functional as F 
import matplotlib.pyplot as plt
# import seaborn as sns
import os

def load_image(filename, mode='RGB'):
    return Image.open(filename).convert(mode)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def visualize_batch(dataloader, num_samples=4):
    """
    Visualizes a batch from BasicDataset, splitting concatenated channels
    and displaying filenames above each group.
    """
    # Get one batch
    batch = next(iter(dataloader))
    data, img_files, mask_files = batch
    
    # If patches were parsed, data is [Batch, N_patches, 7, H, W]
    # We will just visualize the first patch of each sample in the batch for clarity
    if data.ndim == 5:
        data = data[:, 0, :, :, :] 
    
    # Setup plotting
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    plt.subplots_adjust(hspace=0.4)

    for i in range(min(num_samples, data.shape[0])):
        # 1. Split the 7 channels: [0:3] Masked, [3:4] Mask, [4:7] GT
        masked_img = data[i, 0:3, :, :].permute(1, 2, 0).cpu().numpy()
        mask = data[i, 3:4, :, :].squeeze().cpu().numpy()
        gt_img = data[i, 4:7, :, :].permute(1, 2, 0).cpu().numpy()

        # Get filenames - handle list of paths from BasicDataset
        img_name = img_files[i][0] if isinstance(img_files[i], list) else img_files[i]
        mask_name = mask_files[i]

        # 2. Plotting
        # Column 1: Masked Image
        axes[i, 0].imshow(np.clip(masked_img, 0, 1))
        axes[i, 0].set_title(f"Image: {img_name}", fontsize=10)
        axes[i, 0].axis('off')

        # Column 2: Mask Image
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Mask: {mask_name}", fontsize=10)
        axes[i, 1].axis('off')

        # Column 3: Original (GT) Image
        axes[i, 2].imshow(np.clip(gt_img, 0, 1))
        axes[i, 2].set_title("Ground Truth", fontsize=10)
        axes[i, 2].axis('off')

    plt.show()

class ArtPaintingDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, patch_size, n, transforms,
                 scale: float = 1.0, img_size=None, mask_suffix: str = '', split = 'train',
                 n_images = None, n_subsets: int = 50, class_names: list = [], 
                 parse_patches = True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.img_size = [img_size, img_size]
        self.mask_suffix = mask_suffix
        self.class_names = class_names
        self.n_images = n_images
        self.n_subsets = n_subsets
        self.parse_patches = parse_patches
        self.patch_size = patch_size
        self.n = n
        self.transforms = transforms
        # print(Path(images_dir), Path(self.mask_dir))
        self.ids = [
            splitext(str(p.relative_to(images_dir)))[0] 
            for p in Path(images_dir).rglob('*') 
            if p.is_file() and not p.name.startswith('.')
        ]
        self.texture_masks = {}
        for cls in self.class_names:
            cls_path = Path(self.mask_dir) / cls
            if cls_path.is_dir():
                self.texture_masks[cls] = sorted(list(cls_path.glob('*.*'))) #currently doesnt check for hidden files
            else:
                print(f'Warning: Class folder {cls_path} does not exist in mask directory.')

        if split == 'train':
            self.total_units = len(self.class_names) * self.n_subsets
        else:
            self.total_units = sum([len(self.texture_masks[cls]) for cls in self.class_names])

        self.shuffled_ids = list(range(len(self.class_names))) #check back to see if it samples images from the classes

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.split = split

    def __len__(self):
        if self.split == 'train':
            return len(self.ids) * self.total_units
        else:
            return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, img_size=None):
        w, h = pil_img.size
        if img_size is not None:
            newW, newH = img_size
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        
        pil_img = pil_img.resize((newW, newH), resample=Image.Resampling.LANCZOS)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[..., np.newaxis]

        if (img > 1).any():
            img = img / 255.0

        return img

    @staticmethod
    def overlay_mask_on_image(img, mask):
        newW, newH = img.size
        alpha_channel = np.array(mask.resize((newW, newH), resample=Image.Resampling.LANCZOS)).astype(np.float32)[..., np.newaxis]/255.0

        # Combine the original image with the new alpha channel
        orig_img = np.array(img).astype(np.float32)
        img_rgb = np.ones_like(orig_img)*alpha_channel + orig_img*(1-alpha_channel)
        
        return Image.fromarray(img_rgb.astype(np.uint8))
    
    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        #modified for numpy cropping
        crops = []
        for i in range(len(x)):
            new_crop = img[x[i] : x[i] + h, y[i] : y[i] + w, ...]#img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def __getitem__(self, idx):
        img_idx = idx // self.total_units # because getitem is called len(ids)*total_units times
        name = self.ids[img_idx]
        img_file = list(self.images_dir.glob(name + '.*'))#image folder

        if self.split == 'train':
            mask_idx = idx % self.total_units
            cls_idx = mask_idx %len(self.class_names)
            subset_idx = mask_idx // len(self.class_names)
            tgt_cls = self.class_names[self.shuffled_ids[cls_idx]]
            available_masks = self.texture_masks[tgt_cls]
            mask_file = available_masks[subset_idx % len(available_masks)]
        else:
            cls_idx = img_idx % len(self.class_names)#to represent the classes per image
            cls_name = self.class_names[cls_idx]
            available_masks = self.texture_masks[cls_name]
            mask_idx = img_idx % len(available_masks)
            mask_file = available_masks[mask_idx]
            

        mask = load_image(mask_file, mode='L')
        img = load_image(img_file[0])
        masked = self.overlay_mask_on_image(img, mask)

        if not self.parse_patches:
            wd_new, ht_new = self.img_size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            self.img_size = wd_new, ht_new

        img = self.preprocess(img, self.scale, self.img_size)
        masked = self.preprocess(masked, self.scale, self.img_size)
        mask = self.preprocess(mask, self.scale, self.img_size)
        
        if self.parse_patches:#train or val
            i, j, h, w = self.get_params(masked, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(masked, i, j, h, w)
            gt_img = self.n_random_crops(img, i, j, h, w)
            mask_img = self.n_random_crops(mask, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(mask_img[i]),
                                  self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), str(img_file), str(mask_file)
        else:
            return torch.cat([self.transforms(masked), self.transforms(mask), self.transforms(img)], dim=0), str(img_file), str(mask_file)

class ArtPainting:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    def get_loader(self, parse_patches = True, validation = 'art_painting'):
        train_ds = ArtPaintingDataset(images_dir = os.path.join(self.config.data.dataset_path),
                                        mask_dir= os.path.join(self.config.data.mask_path),
                                        patch_size=self.config.data.patch_size,
                                        n=self.config.training.patch_n,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        img_size=self.config.data.image_size,
                                        n_subsets=self.config.data.n_subsets,
                                        class_names=self.config.data.class_names)
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size = self.config.training.batch_size,
                                                   shuffle = True, num_workers = self.config.data.num_workers,
                                                   )#pin_memory = True)
        return train_loader

if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        import yaml
        import os
        import argparse
        def dict2namespace(config):
            namespace = argparse.Namespace()
            for key, value in config.items():
                if isinstance(value, dict):
                    new_value = dict2namespace(value)
                else:
                    new_value = value
                setattr(namespace, key, new_value)
            return namespace
        with open(os.path.join('../configs/art_painting.yml'), "r") as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        train_dl = ArtPainting(new_config).get_loader()
        visualize_batch(train_dl)
        