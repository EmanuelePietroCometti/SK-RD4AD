"""
export_onnx_from_checkpoint.py — Export an SK-RD4AD checkpoint to an ONNX graph
implementing the repo's CANONICAL anomaly definition (contract 2.0).

Canonical pipeline (single source of truth: test.py)
----------------------------------------------------
dynamic crop (host, on the [0,1] image) -> sum of per-layer (1 - cosine
similarity) maps, bilinear upsample align_corners=False -> Gaussian blur k=15
sigma=4 zero padding (IN-GRAPH, kernel imported from test.py) -> score = max of
the BLURRED map (IN-GRAPH).

Contract 2.0 vs 1.0
-------------------
Contract 1.0 exported a raw (un-blurred) map and expected the host to blur it
before scoring; the blur parameters lived in three inconsistent copies
(test.py sigma=4 zero-pad, eval.py cv2 sigma~2.6 reflect, runtime metadata).
Contract 2.0 bakes the one canonical blur into the graph, so "anomaly_score"
is directly comparable with the calibrated threshold and the host does ZERO
post-processing. A runtime still applying a host-side blur on top of this
graph would blur twice: score_source="graph" in the metadata exists precisely
so old runtimes fail loudly instead of doing that silently.

I/O contract
------------
    Input   "input_tensor"  : float32 [B, 3, 256, 256]  host-resized,
            dynamic-cropped (on the [0,1] image) and ImageNet-normalized.
    Output  "anomaly_map"   : float32 [B, 1, 256, 256]  canonical BLURRED map.
    Output  "anomaly_score" : float32 [B]  max of the blurred map — threshold
            this directly against the calibrated threshold.

Only the batch axis is dynamic (``dynamic_axes``).

The graph is always fp32. Reduced precision (fp16/int8) is intentionally NOT
produced here: the inference program casts the fp32 graph to its target
precision, so a single canonical fp32 artifact stays the source of truth.
(If you deploy in fp16/int8, re-run calibrate_threshold.py in that precision.)

Usage
-----
    python export_onnx_from_checkpoint.py ckpt.pth out.onnx
    python export_onnx_from_checkpoint.py --self_test        # random weights
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
OPSET = 17
INPUT_NAMES = ["input_tensor"]
OUTPUT_NAMES = ["anomaly_map", "anomaly_score"]


class RD4ADPure(nn.Module):
    """Pure forward pass: normalized tensor in -> (raw anomaly_map, raw score).

    ``res`` must equal the --res the checkpoint was trained with (it selects the
    decoder's skip-connection structure; see DEFAULT_RES comment above).
    """

    def __init__(self, encoder, bn, decoder, res: int = DEFAULT_RES):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        self.res = res
        # Canonical Gaussian blur (test.py: k=15, sigma=4, zero padding) as a
        # fixed buffer so it is exported as a graph initializer.
        self.register_buffer("gauss_kernel", get_gaussian_kernel(torch.device("cpu")))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_tensor: torch.Tensor):
        feats = self.encoder(input_tensor)
        recon = self.decoder(self.bn(feats), feats[0:3], self.res)

        anomaly_map = None
        for a, b in zip(feats[0:3], recon):
            a_n = F.normalize(a, p=2, dim=1)
            b_n = F.normalize(b, p=2, dim=1)
            dist = 1.0 - torch.sum(a_n * b_n, dim=1, keepdim=True)
            dist = F.interpolate(dist, size=(IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            anomaly_map = dist if anomaly_map is None else anomaly_map + dist

        # Canonical blur baked into the graph, then score = max of the BLURRED
        # map: this is the number the calibrated threshold applies to, with no
        # host-side post-processing. squeeze to [B] for a uniform score
        # signature across all 4 architectures.
        anomaly_map = F.conv2d(anomaly_map, self.gauss_kernel,
                               padding=GAUSS_KERNEL_SIZE // 2)
        score = torch.amax(anomaly_map, dim=(2, 3)).squeeze(1)
        return anomaly_map, score


def build_model(checkpoint_path: str | None, device: str, res: int = DEFAULT_RES) -> RD4ADPure:
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

    model = RD4ADPure(encoder.eval(), bn.eval(), decoder.eval(), res=res).to(device).eval()
    return model


# Embedded in the .onnx file so the inference runtime (inference_simulation) can
# auto-configure itself instead of relying on CLI flags the operator has to
# remember.
#
# score_source="graph" (contract 2.0): the Gaussian blur is baked INTO the
# graph, so "anomaly_score" is the max of the already-blurred map and is
# directly comparable with the calibrated threshold. The runtime must NOT
# apply any host-side blur (doing so would blur twice) and must NOT recompute
# the score from the map. A runtime that only understands contract 1.0
# ("map_max_blurred") must treat this value as an error and refuse to run,
# not silently fall back.
#
# dynamic_crop=true: main.py's training loop ALWAYS applies
# apply_dynamic_crop_gpu (test.py) before the encoder - it crops to the
# bounding box of non-background pixels and rescales it to fill the frame, to
# normalize object scale. The crop operates on the DENORMALIZED [0,1] image
# (its 0.94 background threshold is defined there); host order is:
# resize 256 -> scale to [0,1] -> dynamic crop -> ImageNet normalize -> graph.
# Skipping it (or cropping the normalized image) feeds the model a different
# framing than it was trained on - uniformly bad reconstruction (raw scores
# stuck near their maximum for every image) is the exact signature.
#
# Threshold: eval.py now applies the identical canonical pipeline (crop +
# in-graph blur definition), so its calibration JSON agrees with this graph up
# to floating-point drift. The authoritative number to ship, though, is the
# one from calibrate_threshold.py in this repo: it computes the threshold by
# running THIS .onnx file, so preprocessing and graph are the production ones
# by construction. Cross-check the two; embed with --embed.
EXPORT_METADATA = {
    "anomaly_export_contract": "2.0",
    "architecture": "sk_rd4ad",
    "score_source": "graph",
    "map_blur": "baked_in_graph",
    "blur_kernel_size": "15",
    "blur_sigma": "4.0",       # test.py canonical kernel (training model selection)
    "blur_padding": "zeros",
    "dynamic_crop": "true",
    "dynamic_crop_bg_threshold": "0.94",
    "dynamic_crop_padding": "30",
    "dynamic_crop_input_range": "0_1_before_normalization",
    "verified": "false",       # contract 2.0 not yet e2e-revalidated on Colab
                               # (contract 1.0 was, 2026-07-09: real checkpoint,
                               # auc=0.91, defect correctly localized). Graph
                               # parity is checked at export time by verify();
                               # run parity_check.py + calibrate_threshold.py,
                               # then flip this after the e2e rerun.
}


def _write_metadata(onnx_path: Path, weights_source: str, res: int) -> None:
    """weights_source: "checkpoint:<filename>" for real exports, or
    "random_self_test" for --self_test exports. The inference runtime refuses
    to score with a random_self_test model: with a random decoder the cosine
    distance saturates (~uniform map, near-constant max score for EVERY image
    -> all-red heatmaps, "anomaly score = 1"), which looks like a subtle
    pipeline bug instead of what it is - a model with no trained weights.

    res: the skip-connection setting baked into this graph; recorded so it is
    always possible to check it against the training run's --res afterwards."""
    import onnx
    m = onnx.load(str(onnx_path))
    for k, v in {**EXPORT_METADATA, "weights_source": weights_source, "res": str(res)}.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, str(onnx_path))


def export_fp32(model: RD4ADPure, onnx_path: Path, device: str, weights_source: str):
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
    dynamic_axes = {
        "input_tensor": {0: "batch"},
        "anomaly_map": {0: "batch"},
        "anomaly_score": {0: "batch"},
    }
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(onnx_path),
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes, opset_version=OPSET,
            do_constant_folding=True, dynamo=False, external_data=False,
        )
    import onnx
    onnx.checker.check_model(str(onnx_path))
    _write_metadata(onnx_path, weights_source, model.res)


def verify(model, onnx_path, device, atol=1e-3, rtol=1e-3):
    # PyTorch and ONNX Runtime use different conv/reduction kernels, so bit-exact
    # parity is impossible on a network this deep; expect ~1e-4 drift on the raw
    # map. Tolerances are set to catch real bugs (wrong op, transposed weights)
    # while allowing normal floating-point divergence.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    print(f"\n--- PyTorch vs ONNX parity (atol={atol:g}, rtol={rtol:g}) ---")
    for batch in (1, 4):
        x = rng.standard_normal((batch, 3, IMG_SIZE, IMG_SIZE)).astype(np.float32)
        with torch.no_grad():
            tm, ts = model(torch.from_numpy(x).to(device))
        om, os_ = sess.run(OUTPUT_NAMES, {INPUT_NAMES[0]: x})
        np.testing.assert_allclose(om, tm.cpu().numpy(), atol=atol, rtol=rtol,
                                   err_msg=f"anomaly_map mismatch (batch={batch})")
        np.testing.assert_allclose(os_, ts.cpu().numpy(), atol=atol, rtol=rtol,
                                   err_msg=f"anomaly_score mismatch (batch={batch})")
        print(f"  batch={batch}: map |Δ|max={np.abs(om-tm.cpu().numpy()).max():.2e}  "
              f"score |Δ|max={np.abs(os_-ts.cpu().numpy()).max():.2e}  OK")
    print("[PASS] Numerical parity within tolerance confirmed.")


def resolve_output_path(output: str, checkpoint: str | None) -> Path:
    """Turn the ``output`` argument into a concrete ``.onnx`` file path.

    Accepts either a full file path (``.../model.onnx``) or a directory. A
    directory is detected if the path already exists as one, or has no ``.onnx``
    suffix (so ``exports`` or ``exports/`` both mean "put the file in here"); in
    that case the filename is derived from the checkpoint stem. This avoids the
    ``PermissionError``/``IsADirectoryError`` from handing torch.onnx.export a
    directory to open for writing.
    """
    out = Path(output)
    is_dir = out.is_dir() or out.suffix.lower() != ".onnx"
    if is_dir:
        stem = Path(checkpoint).stem if checkpoint else "sk_rd4ad_selftest"
        out = out / f"{stem}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to the .pth checkpoint ('bn' + 'decoder'). Omit with --self_test.")
    p.add_argument("output", nargs="?", default="sk_rd4ad_selftest.onnx",
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
        # With --self_test there is no checkpoint to speak of, so a single
        # positional argument means "output path", not "checkpoint" (argparse
        # otherwise binds it to the first positional declared).
        if args.checkpoint is not None and args.output == "sk_rd4ad_selftest.onnx":
            args.output = args.checkpoint
        elif args.checkpoint is not None:
            # A previous version only warned here and exported RANDOM weights
            # anyway. That produced a production-looking .onnx whose decoder
            # reconstructs nothing: anomaly map uniformly ~1 (all-red heatmap)
            # and a near-constant max score for every image - symptoms that
            # look like a pipeline bug rather than untrained weights. Refuse
            # instead: the two options are mutually exclusive on purpose.
            sys.exit(
                "ERROR: --self_test and a checkpoint are mutually exclusive.\n"
                "  - To export your TRAINED model:  python export_onnx_from_checkpoint.py "
                "<checkpoint.pth> <output>\n"
                "  - To test the export pipeline with random weights:  "
                "python export_onnx_from_checkpoint.py --self_test <output>\n"
                "A --self_test model must never be used for real inference: its random "
                "decoder saturates the anomaly map (~1 everywhere, all-red heatmaps, "
                "identical scores for every image)."
            )
        ckpt = None
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            sys.exit("ERROR: provide a checkpoint path, or pass --self_test for a random-weight export.")
        if not Path(ckpt).is_file():
            sys.exit(f"Checkpoint not found: {ckpt}")

    model = build_model(ckpt, device, res=args.res)
    out_path = resolve_output_path(args.output, ckpt)
    weights_source = f"checkpoint:{Path(ckpt).name}" if ckpt else "random_self_test"

    print(f"\n--- Exporting SK-RD4AD (contract 2.0: blur+score in-graph, weights: {weights_source}, res={args.res}) -> {out_path} ---")
    export_fp32(model, out_path, device, weights_source)
    print(f"[OK] fp32 export: {out_path}")

    # Verify for BOTH self-test and real exports (parity holds regardless of the
    # weights); only skip if the user explicitly opts out.
    if args.self_test or not args.no_verify:
        verify(model, out_path, device)


if __name__ == "__main__":
    main()
