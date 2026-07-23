"""
export_onnx_from_checkpoint.py — Export an SK-RD4AD checkpoint to an ONNX graph
under the shared export contract 3.0 (see export_common.py, identical in the
SuperSimpleNet / anomalib / SK-RD4AD repos).

Contract 3.0 in one paragraph
-----------------------------
    Input   "image"         : float32 [B, 3, 256, 256]  RGB in [0,1], model
            resolution, host-resized and dynamic-cropped, NOT normalized —
            ImageNet normalization is IN-GRAPH (InGraphNormalize).
    Output  "anomaly_map"   : float32 [B, 1, 256, 256]  FINAL map: sum of
            per-layer (1 - cosine) maps, bilinear upsample align_corners=False,
            canonical Gaussian blur k=15 sigma=4 zero-pad, all IN-GRAPH.
    Output  "anomaly_score" : float32 [B]  max of the BLURRED map (IN-GRAPH),
            directly comparable with metadata["calibrated_threshold"].

Only the batch axis is dynamic.

Migration from contract 2.0 (this file's previous version)
----------------------------------------------------------
Contract 2.0 kept ImageNet normalization on the host and named the graph input
``input_tensor``; the metadata used a bespoke 2.0 schema (``score_source`` /
``map_blur`` / ``anomaly_export_contract``). Under 3.0 the four architectures
share ONE contract: normalization is a graph node, the input is plain RGB in
[0,1], the input is named ``image``, and the metadata follow ``build_metadata``.
The anomaly math (cosine maps + upsample + canonical blur + amax score) is
UNCHANGED — it now lives inside ``core`` of an ``ExportWrapper`` subclass, with
``InGraphNormalize`` prepended by the base ``forward``.

The dynamic crop stays a HOST step (as it does for the input production of every
architecture): it is data-dependent and operates on the DENORMALIZED [0,1] image
(0.94 background threshold). It is NOT in the graph; ``dynamic_crop=true`` plus
the ``dynamic_crop_*`` metadata keys tell the runtime it must apply it. Required
host order:  resize 256 -> scale to [0,1] -> dynamic crop -> graph (which
normalizes internally). See CONTRACT.md 2.7.

The blur (k=15, sigma=4, ZERO padding) is SK-RD4AD's motivated exception to the
otherwise-shared blur: it is the canonical kernel of test.py, the same object
used by eval.py and by training-time model selection. Never redefine it here.

The graph is always fp32. Reduced precision (fp16/int8) is intentionally NOT
produced here: the inference program casts the fp32 graph to its target
precision. (If you deploy in fp16/int8, re-run calibrate_threshold.py there.)

Usage
-----
    python export_onnx_from_checkpoint.py ckpt.pth out.onnx
    python export_onnx_from_checkpoint.py --self_test        # random weights
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from export_common import (
    ExportWrapper,
    assert_trained_bn,
    build_metadata,
    export,
    resolve_output_path,
    verify,
)
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
# The Gaussian kernel is imported from test.py on purpose: it is the SAME
# object used by eval.py and by training-time model selection. Never redefine
# it here — that is exactly the training/inference drift this export prevents.
from test import GAUSS_KERNEL_SIZE, get_gaussian_kernel

# Default number-of-skip-connections parameter (main.py --res default). MUST
# match the value the checkpoint was TRAINED with: it changes the decoder's
# forward structure (which skip connections are added), so a mismatch produces
# a graph that computes something the trained weights never saw - garbage
# output with no error raised. Override with --res at export time.
DEFAULT_RES = 3
IMG_SIZE = 256


class SKRD4ADExport(ExportWrapper):
    """Contract-3.0 wrapper for SK-RD4AD.

    ``ExportWrapper.forward`` prepends ``InGraphNormalize`` (the [0,1] RGB input
    becomes ImageNet-normalized inside the graph) and canonicalizes shapes.
    ``core`` is the repo's canonical anomaly pipeline, unchanged:

        encoder -> bn -> decoder -> sum of per-layer (1 - cosine) maps ->
        bilinear upsample align_corners=False -> canonical Gaussian blur
        (k=15, sigma=4, ZERO padding, kernel from test.py) -> score = amax.

    ``res`` must equal the --res the checkpoint was trained with (it selects the
    decoder's skip-connection structure; see DEFAULT_RES).
    """

    def __init__(self, encoder, bn, decoder, res: int = DEFAULT_RES):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        self.res = res
        # Canonical blur baked as a graph initializer (fixed buffer).
        self.register_buffer("gauss_kernel", get_gaussian_kernel(torch.device("cpu")))
        for p in self.parameters():
            p.requires_grad_(False)

    def core(self, x: torch.Tensor):
        feats = self.encoder(x)
        recon = self.decoder(self.bn(feats), feats[0:3], self.res)

        anomaly_map = None
        for a, b in zip(feats[0:3], recon):
            a_n = F.normalize(a, p=2, dim=1)
            b_n = F.normalize(b, p=2, dim=1)
            dist = 1.0 - torch.sum(a_n * b_n, dim=1, keepdim=True)
            dist = F.interpolate(dist, size=(IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            anomaly_map = dist if anomaly_map is None else anomaly_map + dist

        anomaly_map = F.conv2d(anomaly_map, self.gauss_kernel,
                               padding=GAUSS_KERNEL_SIZE // 2)
        score = torch.amax(anomaly_map, dim=(2, 3))  # [B,1]; ExportWrapper -> [B]
        return anomaly_map, score


def build_model(checkpoint_path: str | None, device: str,
                res: int = DEFAULT_RES) -> SKRD4ADExport:
    # For --self_test we skip the pretrained download; parity does not need
    # trained weights (it compares the two graphs, not accuracy).
    encoder, bn = wide_resnet50_2(pretrained=checkpoint_path is not None)
    decoder = de_wide_resnet50_2(pretrained=False)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if not (isinstance(ckpt, dict) and "decoder" in ckpt and "bn" in ckpt):
            sys.exit("ERROR: checkpoint must be a dict with 'bn' and 'decoder' keys.")
        decoder.load_state_dict(ckpt["decoder"])
        bn.load_state_dict(ckpt["bn"])
        # Trained-state guard: a decoder still at init (num_batches_tracked==0
        # in every BatchNorm) reconstructs nothing — the cosine distance
        # saturates, the map is ~uniform and the max score is ~constant for
        # every image (all-red heatmaps). That looks like a pipeline bug, not
        # untrained weights. Refuse. --self_test is the explicit escape hatch
        # (it marks weights_source=random_self_test, which the runtime rejects).
        assert_trained_bn(
            decoder,
            "SK-RD4AD export (decoder loaded from "
            f"checkpoint:{Path(checkpoint_path).name})",
        )

    model = SKRD4ADExport(encoder.eval(), bn.eval(), decoder.eval(), res=res)
    return model.to(device).eval()


def _skrd4ad_metadata(weights_path: Path | None, res: int) -> dict:
    """Contract-3.0 metadata for SK-RD4AD.

    dynamic_crop=true plus the dynamic_crop_* extras tell the runtime it must
    apply the host-side crop (test.py apply_dynamic_crop_gpu) on the [0,1]
    image BEFORE the graph; the graph normalizes internally. blur_padding=zeros
    records SK-RD4AD's motivated exception to the shared blur. res records the
    baked skip-connection structure so it can be cross-checked against training.
    """
    return build_metadata(
        architecture="sk_rd4ad",
        image_size=(IMG_SIZE, IMG_SIZE),
        blur_kernel_size=GAUSS_KERNEL_SIZE,        # 15, test.py canonical
        blur_sigma=4.0,                            # test.py canonical
        dynamic_crop=True,
        weights_path=weights_path,
        resize_mode="bilinear_antialias",          # dataset/dataset.py v2.Resize
        extra={
            "blur_padding": "zeros",               # exception vs SSN reflect pad
            "dynamic_crop_bg_threshold": "0.94",
            "dynamic_crop_padding": "30",
            "dynamic_crop_input_range": "0_1_before_normalization",
            "res": res,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to the .pth checkpoint ('bn' + 'decoder'). Omit with --self_test.")
    p.add_argument("output", nargs="?", default=None,
                   help="Output .onnx file, or a directory to place it in.")
    p.add_argument("--self_test", action="store_true",
                   help="Export with RANDOM weights to test the export pipeline itself "
                        "(NOT usable for real inference: a random decoder saturates the "
                        "anomaly map at ~1 for every image). Incompatible with a checkpoint.")
    p.add_argument("--no_verify", action="store_true",
                   help="Skip the PyTorch<->ONNX parity check after a real export.")
    p.add_argument("--res", type=int, default=DEFAULT_RES, choices=[1, 2, 3],
                   help="MUST match the --res used at TRAINING time (default: 3, the "
                        "training default). Selects the decoder's skip-connection "
                        "structure; a mismatch silently produces garbage output.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.self_test:
        # With --self_test there is no checkpoint, so a single positional
        # argument means "output path", not "checkpoint".
        if args.checkpoint is not None and args.output is None:
            args.output = args.checkpoint
        elif args.checkpoint is not None:
            # Refuse rather than silently exporting random weights behind a
            # production-looking filename: a random decoder reconstructs nothing
            # (anomaly map ~1 everywhere, identical scores for every image).
            sys.exit(
                "ERROR: --self_test and a checkpoint are mutually exclusive.\n"
                "  - To export your TRAINED model:  python export_onnx_from_checkpoint.py "
                "<checkpoint.pth> <output>\n"
                "  - To test the export pipeline with random weights:  "
                "python export_onnx_from_checkpoint.py --self_test <output>"
            )
        ckpt = None
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            sys.exit("ERROR: provide a checkpoint path, or pass --self_test for a random-weight export.")
        if not Path(ckpt).is_file():
            sys.exit(f"Checkpoint not found: {ckpt}")

    weights = Path(ckpt) if ckpt else None
    model = build_model(ckpt, device, res=args.res)
    out_path = resolve_output_path(args.output, weights, "sk_rd4ad_selftest")
    metadata = _skrd4ad_metadata(weights, args.res)

    print(f"\n--- Exporting SK-RD4AD (contract 3.0, weights: {metadata['weights_source']}, "
          f"res={args.res}) -> {out_path} ---")
    export(model, (IMG_SIZE, IMG_SIZE), out_path, device, metadata)
    print(f"[OK] fp32 export: {out_path}")
    print(f"     input  'image'         : float32 [B,3,{IMG_SIZE},{IMG_SIZE}] RGB [0,1] "
          "(normalization in-graph; host must dynamic-crop before the graph)")
    print( "     output 'anomaly_map'   : float32 [B,1,H,W] (final: upsample + blur in-graph)")
    print( "     output 'anomaly_score' : float32 [B] (final: amax of blurred map)")
    print( "     NOTA: verified=false — eseguire calibrate_threshold.py per scrivere "
           "calibrated_threshold/calibration_global_min/max e verified=true.")

    # Verify for BOTH self-test and real exports (parity holds regardless of the
    # weights); only skip if the user explicitly opts out.
    if args.self_test or not args.no_verify:
        verify(model, out_path, (IMG_SIZE, IMG_SIZE), device)


if __name__ == "__main__":
    main()
