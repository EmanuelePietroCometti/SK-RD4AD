"""Caricamento di un backbone fine-tuned come teacher di SK-RD4AD.

Accetta lo state_dict puro, un dict di training (chiavi 'encoder', 'state_dict',
'model_state_dict', ...) come il best_encoder.pth del repo transfer_learning,
o un nn.Module pickled.
"""
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Custom teacher support
# ---------------------------------------------------------------------------
# Prefixes added by common wrappers (DataParallel, Lightning, custom
# `self.backbone = resnet18(...)` classifiers). Stripped until the key matches.
_WRAPPER_PREFIXES = ('module.', 'backbone.', 'encoder.', 'model.', 'net.', 'feature_extractor.')


def _extract_state_dict(obj):
    """Accept a raw state_dict, a pickled nn.Module or a training checkpoint dict."""
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ('encoder', 'state_dict', 'model_state_dict', 'model', 'net'):
            if isinstance(obj.get(key), (dict, nn.Module)):
                return _extract_state_dict(obj[key])
        return obj
    raise TypeError(f'Unsupported checkpoint type: {type(obj).__name__}')


def _is_optional(key: str) -> bool:
    # fc: replaced by the classification head during fine-tuning, never used here.
    # layer4: computed in forward() but its output is discarded by SK-RD4AD.
    return key.startswith(('fc.', 'layer4.')) or key.endswith('num_batches_tracked')


def load_custom_encoder(encoder: nn.Module, ckpt_path: str) -> nn.Module:
    """Overwrite the ImageNet weights of `encoder` with a fine-tuned backbone.

    Fails loudly if any tensor consumed by SK-RD4AD (stem, layer1-3) is missing
    or has the wrong shape: a silent partial load would train the student
    against an (almost) ImageNet teacher without any visible error.
    """
    try:
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        # Full pickled nn.Module (torch.save(model)). Executes pickle code:
        # only load files you produced yourself.
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    target = encoder.state_dict()
    loaded = {}
    for k, v in _extract_state_dict(raw).items():
        nk = k
        while nk not in target:
            prefix = next((p for p in _WRAPPER_PREFIXES if nk.startswith(p)), None)
            if prefix is None:
                break
            nk = nk[len(prefix):]
        loaded[nk] = v

    matched = {k: v for k, v in loaded.items()
               if k in target and not k.startswith('fc.')}
    ignored = sorted(k for k in loaded if k not in target)

    # Backbone tensors the encoder has no slot for => different depth (e.g. a
    # resnet34 checkpoint loaded into res18 would otherwise load partially).
    extra = [k for k in ignored if k.split('.')[0] in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')]
    if extra:
        raise ValueError(f'[custom encoder] checkpoint has backbone tensors not in --net: {extra[:5]}')

    bad_shape = [f'{k}: {tuple(v.shape)} vs {tuple(target[k].shape)}'
                 for k, v in matched.items() if v.shape != target[k].shape]
    if bad_shape:
        raise ValueError(f'[custom encoder] shape mismatch (wrong --net?): {bad_shape[:5]}')

    missing = [k for k in target if k not in matched and not _is_optional(k)]
    if missing:
        raise KeyError(f'[custom encoder] {len(missing)} required tensors missing, e.g. '
                       f'{missing[:5]}. Unmatched keys in checkpoint, e.g. {ignored[:5]}')

    encoder.load_state_dict(matched, strict=False)
    if not any(k.startswith('layer4.') for k in matched):
        print('[custom encoder] layer4 not in checkpoint: kept ImageNet weights (unused downstream)')
    print(f'[custom encoder] loaded {len(matched)} tensors from {ckpt_path} '
          f'({len(ignored)} ignored, e.g. {ignored[:3]})')
    return encoder
