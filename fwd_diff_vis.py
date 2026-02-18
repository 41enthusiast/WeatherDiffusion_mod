import torch
import numpy as np
from PIL import Image

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_noised_image(x_0, t, betas):
    """
    x_0: original image tensor (B, C, H, W) in range [-1, 1]
    t: timestep index
    betas: the beta schedule
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])
    
    noise = torch.randn_like(x_0)
    
    # Apply the forward diffusion formula
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

# --- Setup ---
T = 100  # Total timesteps for the GIF
betas = linear_beta_schedule(T)

# Load your image, resize to config.patch_size, and normalize to [-1, 1]
img = Image.open("../art_painting_data/test/image/0ed9b4037a43a881b17ae85322e7ef8a.png").convert("RGB").resize((64, 64))
x_0 = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
x_0 = x_0.unsqueeze(0) # Add batch dimension

frames = []
for t in range(T):
    with torch.no_grad():
        # Get noisy image at step t
        x_t = get_noised_image(x_0, torch.tensor([t]), betas)
        
        # Convert back to [0, 255] for PIL
        out = ((x_t.squeeze().permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        frames.append(Image.fromarray(out))

# Save as GIF
frames[0].save('misc/diffusion_noising.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)