"""Configurazione della definizione del punteggio di anomalia di SK-RD4AD.

Stessa filosofia di aug_config.py: ogni scelta che incide sul punteggio (e quindi su AUROC,
soglie e calib.json) e' un campo esplicito, con default uguali al codice degli autori
(https://github.com/pej0918/SK-RD4AD).

Due preset:

  "paper"      Definizione del repo degli autori: mappa = somma dei (1 - coseno) dei tre livelli,
               upsampling bilineare con align_corners=True, poi
               scipy.ndimage.gaussian_filter(sigma=4) (kernel 33x33, bordi riflessi),
               punteggio d'immagine = massimo della mappa sfocata. Nessun crop in valutazione.
               E' il default.

  "canonical"  Definizione usata finora in questo fork (adatta all'export ONNX): kernel 15x15 con
               zero-padding, align_corners=False. Va usata finche' l'exporter ONNX e il runtime
               incorporano il blur 15x15: le soglie in calib.json sono valide solo per il punteggio
               con cui sono state calcolate.

Il crop dinamico in valutazione e' separato dal preset: con dynamic_crop=None segue
AugConfig.dynamic_crop, cosi' training e valutazione usano lo stesso preprocessing.
"""
import json
from dataclasses import dataclass, asdict, fields
from typing import Optional

EVAL_SCHEMA_VERSION = "1.0"

VALID_PADDING = ("symmetric", "zeros")  # 'symmetric' == scipy mode='reflect' (default di gaussian_filter)

# Campi che definiscono il punteggio. dynamic_crop e' escluso: e' preprocessing, non preset.
EVAL_PRESETS = {
    "paper": dict(blur_sigma=4.0, blur_kernel_size=None, blur_padding="symmetric", align_corners=True),
    "canonical": dict(blur_sigma=4.0, blur_kernel_size=15, blur_padding="zeros", align_corners=False),
}


@dataclass
class EvalConfig:
    blur_sigma: float = 4.0
    # None => dimensione di scipy.ndimage.gaussian_filter con truncate=4.0: 2*int(4*sigma + 0.5) + 1
    # (33 per sigma=4). Altrimenti un intero dispari.
    blur_kernel_size: Optional[int] = None
    blur_padding: str = "symmetric"   # 'symmetric' (= scipy 'reflect') | 'zeros'
    align_corners: bool = True        # upsampling bilineare delle mappe per livello
    # None => segue AugConfig.dynamic_crop (training e valutazione coerenti); True/False forza.
    dynamic_crop: Optional[bool] = None

    def __post_init__(self):
        if self.blur_sigma <= 0:
            raise ValueError(f"blur_sigma deve essere > 0, ricevuto {self.blur_sigma}")
        if self.blur_padding not in VALID_PADDING:
            raise ValueError(f"blur_padding deve essere uno tra {VALID_PADDING}, ricevuto {self.blur_padding!r}")
        if self.blur_kernel_size is not None and (self.blur_kernel_size < 1 or self.blur_kernel_size % 2 == 0):
            raise ValueError(f"blur_kernel_size deve essere un intero dispari >= 1, ricevuto {self.blur_kernel_size}")

    # ------------------------------------------------------------------ costruzione
    @classmethod
    def from_preset(cls, name: str, **overrides) -> "EvalConfig":
        if name not in EVAL_PRESETS:
            raise ValueError(f"preset sconosciuto {name!r}; disponibili: {sorted(EVAL_PRESETS)}")
        return cls(**{**EVAL_PRESETS[name], **overrides})

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        data = dict(data)
        data.pop("schema_version", None)
        data.pop("preset", None)  # solo informativo in to_dict()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"chiavi sconosciute in EvalConfig: {unknown}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ derivati
    def kernel_size(self) -> int:
        """Dimensione effettiva del kernel gaussiano."""
        if self.blur_kernel_size is not None:
            return self.blur_kernel_size
        return 2 * int(4.0 * self.blur_sigma + 0.5) + 1

    def preset_name(self) -> str:
        """'paper', 'canonical' oppure 'custom' (ignora dynamic_crop)."""
        mine = dict(blur_sigma=self.blur_sigma, blur_kernel_size=self.kernel_size(),
                    blur_padding=self.blur_padding, align_corners=self.align_corners)
        for name, p in EVAL_PRESETS.items():
            ref = dict(p)
            if ref["blur_kernel_size"] is None:
                ref["blur_kernel_size"] = 2 * int(4.0 * ref["blur_sigma"] + 0.5) + 1
            if mine == ref:
                return name
        return "custom"

    def resolved(self, aug_cfg=None) -> "EvalConfig":
        """Copia con dynamic_crop risolto in bool (None => AugConfig.dynamic_crop, o False)."""
        crop = self.dynamic_crop
        if crop is None:
            crop = bool(getattr(aug_cfg, "dynamic_crop", False)) if aug_cfg is not None else False
        return EvalConfig(blur_sigma=self.blur_sigma, blur_kernel_size=self.blur_kernel_size,
                          blur_padding=self.blur_padding, align_corners=self.align_corners,
                          dynamic_crop=bool(crop))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blur_kernel_size"] = self.kernel_size()  # esplicito: chi legge non deve ricalcolarlo
        d["preset"] = self.preset_name()
        d["schema_version"] = EVAL_SCHEMA_VERSION
        return d


def build_eval_config(preset: str = "paper", config_path: Optional[str] = None,
                      crop: str = "inherit") -> EvalConfig:
    """Costruisce la config dalla CLI: preset -> eventuale JSON (sovrascrive) -> flag crop.

    crop: 'inherit' (segue AugConfig.dynamic_crop), 'on' o 'off'.
    """
    cfg = EvalConfig.from_preset(preset)
    if config_path:
        cfg = EvalConfig.from_dict({**{k: v for k, v in asdict(cfg).items()}, **_load_json(config_path)})
    if crop not in ("inherit", "on", "off"):
        raise ValueError(f"crop deve essere 'inherit', 'on' o 'off', ricevuto {crop!r}")
    if crop != "inherit":
        cfg = EvalConfig.from_dict({**asdict(cfg), "dynamic_crop": crop == "on"})
    return cfg


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
