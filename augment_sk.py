"""Builds the GPU augmentation Compose from an AugConfig.

Operates on GPU tensors already denormalized to [0,1], AFTER resize and BEFORE
ImageNet renormalization. speckle_std, dynamic_crop and equalize are handled in the
training loop, NOT here.
"""
from torchvision.transforms import v2
from aug_config import AugConfig


def build_gpu_augmentation(cfg: AugConfig):
    """Return a v2.Compose, or None if augmentation is disabled.

    Only families with a non-default value are added, so the Compose contains
    exactly the requested transforms and nothing else.
    """
    if not cfg.enabled:
        return None

    ops = []

    if cfg.hflip_p > 0:
        ops.append(v2.RandomHorizontalFlip(p=cfg.hflip_p))
    if cfg.vflip_p > 0:
        ops.append(v2.RandomVerticalFlip(p=cfg.vflip_p))

    if cfg.affine_deg > 0 or cfg.affine_translate > 0 or tuple(cfg.affine_scale) != (1.0, 1.0):
        ops.append(v2.RandomAffine(
            degrees=[-cfg.affine_deg, cfg.affine_deg],
            translate=[cfg.affine_translate, cfg.affine_translate],
            scale=tuple(cfg.affine_scale),
            fill=1.0,
            interpolation=v2.InterpolationMode.BILINEAR,
        ))

    if cfg.brightness > 0 or cfg.contrast > 0 or cfg.saturation > 0 or cfg.hue > 0:
        ops.append(v2.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        ))

    if cfg.grayscale_p > 0:
        ops.append(v2.RandomGrayscale(p=cfg.grayscale_p))

    if cfg.blur_p > 0:
        ops.append(v2.RandomApply(
            [v2.GaussianBlur(kernel_size=cfg.blur_kernel, sigma=tuple(cfg.blur_sigma))],
            p=cfg.blur_p,
        ))

    return v2.Compose(ops) if ops else None