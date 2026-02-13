from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class DataConfig:
    dataset_name: str = "art_painting_v4"
    dataset_path: str = "../../art_painting_data"
    mask_path: str = "../../dtd/images"
    n_subsets: int = 3
    class_names: Optional[list] = field(default_factory=lambda: [
        'banded', 'blotchy', 'braided', 'bubbly', 'bumpy'])
    img_size: int = 256
    batch_size: int = 16
    num_workers: int = 8
    augment: bool = True

@dataclass
class ModelConfig_ENB0:
    backbone: str = "efficientnet_b0"
    model_name: str = "unet_efb0_730k"
    pretrained: bool = True
    layer1_features: int = 32
    layer2_features: int = 16
    layer3_features: int = 24
    layer4_features: int = 40
    layer5_features: int = 80
    alpha_l: float = 1.0 # weight for L1 loss
    beta_l: float = 0.0#1.0 # weight for perceptual loss
    gamma_l: float = 0.0#150.0 # weight for style loss

@dataclass
class ModelConfig_Swin:
    backbone: str = "swin"
    model_name: str = "unet_swin_41M"
    pretrained: bool = True
    img_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 1
    embed_dim: int = 96
    depths: Optional[list] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: Optional[list] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

# @dataclass
# class TrainConfig:
#     lr: float = 1e-3
#     max_epochs: int = 5000
#     precision: str = 'bf16'        
#     num_devices: int = 1
#     accumulate_grad_batches: int = 1
#     patience: int = 5          # early stopping
#     monitor: str = "val_loss"
#     tr_split: float = 0.8

# def log_dataclass_config(logger, **configs):
#     flat_config = {}
#     for name, cfg in configs.items():
#         flat_config[name] = asdict(cfg)
#     logger.experiment.config.update(flat_config, allow_val_change=True)