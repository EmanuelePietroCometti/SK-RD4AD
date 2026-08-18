"""Ablatable augmentation configuration for SK-RD4AD.

All augmentation is off by default (enabled=False) so that a clean, augmentation-free
baseline is reachable for the first time. Legacy behaviour is reproduced exactly via
configs/aug_legacy.json.
"""
import json
from dataclasses import dataclass, field, asdict

AUG_SCHEMA_VERSION = "1.0"


@dataclass
class AugConfig:
    enabled: bool = False              # master switch: False => NO augmentation at all
    dynamic_crop: bool = False         # apply_dynamic_crop_gpu (was always-on in legacy)
    equalize_p: float = 0.0            # probability of histogram equalization
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    affine_deg: float = 0.0            # +/- degrees; 0 => no rotation
    affine_translate: float = 0.0      # fraction; 0 => no translation
    affine_scale: tuple = (1.0, 1.0)   # (1.0, 1.0) => no scaling
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    grayscale_p: float = 0.0
    blur_p: float = 0.0                # probability of Gaussian blur
    blur_kernel: int = 3
    blur_sigma: tuple = (0.1, 1.0)
    speckle_std: float = 0.0           # additive Gaussian noise std on [0,1]; 0 => off
    seed: int = 0                      # recorded for provenance/logging only (see note)

    @classmethod
    def from_json(cls, path: str) -> "AugConfig":
        with open(path, "r") as f:
            data = json.load(f)
        data.pop("schema_version", None)
        # JSON arrays load as lists; restore tuples for torchvision.
        for k in ("affine_scale", "blur_sigma"):
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = AUG_SCHEMA_VERSION
        return d