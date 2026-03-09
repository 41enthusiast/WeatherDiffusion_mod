import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

def get_masked_image(img, mask):
    newW, newH = img.size
    alpha_channel = np.array(mask.resize((newW, newH), resample=Image.Resampling.LANCZOS)).astype(np.float32)[..., np.newaxis]/255.0

    # Combine the original image with the new alpha channel
    print('Alpha channel range', alpha_channel.min(), alpha_channel.max())
    orig_img = np.array(img).astype(np.float32)/255.0
    img_rgb = np.ones_like(orig_img)*alpha_channel + orig_img*(1-alpha_channel)
    
    return torch.tensor(img_rgb).permute(2,0,1)# / 255.0

gt_file = 'qual_tests/4_clean.png'
mask_file = '../dtd/images/grooved/grooved_0061.jpg'
gt_img = Image.open(gt_file).convert('RGB')
new_width, new_height = gt_img.size
mask_img = Image.open(mask_file).convert('L').resize((new_width, new_height), resample=Image.NEAREST)
mask_t = to_tensor(mask_img)
mask_t[mask_t>0.5]=1
mask_t[mask_t<=0.5]=0
mask_img = to_pil_image(mask_t)
mask_img.save(f"outputs/basic_tests/mask_{mask_file.split('/')[-1]}")
os.makedirs("outputs/basic_tests", exist_ok = True)
masked_img = get_masked_image(gt_img, mask_img)
to_pil_image(masked_img).save(f"outputs/basic_tests/masked_{mask_file.split('/')[-1]}")