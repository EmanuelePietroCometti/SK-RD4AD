"""
export_onnx_from_checkpoint.py — Export an SK-RD4AD checkpoint to a *pure* ONNX
graph (encoder + bottleneck + decoder + cosine-distance anomaly map only).

Why this changed
----------------
The previous export baked resize + ImageNet normalization + Gaussian blur into
the graph. The GPU runtime (``inference_simulation``) normalizes on the host too,
so images were normalized twice and the map blurred twice -> meaningless output.
The graph is now a pure forward pass; all pre/post-processing lives on the host.

Uniform I/O contract (identical for all 4 architectures)
--------------------------------------------------------
    Input   "input_tensor" : float32 [B, 3, 256, 256]  host-resized + ImageNet-
            normalized (the RD4AD encoder expects normalized input).
    Output  "anomaly_map"   : float32 [B, 1, 256, 256]  sum of per-layer
            (1 - cosine similarity) maps, bilinearly upscaled. NO Gaussian blur.
    Output  "anomaly_score" : float32 [B]  max over the *raw* (un-blurred) map.

Only the batch axis is dynamic (``dynamic_axes``). Host post-processing (blur +
folder-global min-max) is unchanged in inference_simulation.

The graph is always fp32. Reduced precision (fp16/int8) is intentionally NOT
produced here: the inference program casts the fp32 graph to its target
precision, so a single canonical fp32 artifact stays the source of truth.

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

RES = 3
IMG_SIZE = 256
OPSET = 17
INPUT_NAMES = ["input_tensor"]
OUTPUT_NAMES = ["anomaly_map", "anomaly_score"]


class RD4ADPure(nn.Module):
    """Pure forward pass: normalized tensor in -> (raw anomaly_map, raw score)."""

    def __init__(self, encoder, bn, decoder):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_tensor: torch.Tensor):
        feats = self.encoder(input_tensor)
        recon = self.decoder(self.bn(feats), feats[0:3], RES)

        anomaly_map = None
        for a, b in zip(feats, recon):
            a_n = F.normalize(a, p=2, dim=1)
            b_n = F.normalize(b, p=2, dim=1)
            dist = 1.0 - torch.sum(a_n * b_n, dim=1, keepdim=True)
            dist = F.interpolate(dist, size=(IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            anomaly_map = dist if anomaly_map is None else anomaly_map + dist

        # raw map -> score = spatial max (no blur). squeeze to [B] for a uniform
        # score signature across all 4 architectures.
        score = torch.amax(anomaly_map, dim=(2, 3)).squeeze(1)
        return anomaly_map, score


def build_model(checkpoint_path: str | None, device: str) -> RD4ADPure:
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

    model = RD4ADPure(encoder.eval(), bn.eval(), decoder.eval()).to(device).eval()
    return model


# Embedded in the .onnx file so the inference runtime (inference_simulation) can
# auto-configure itself instead of relying on CLI flags the operator has to
# remember. score_source="map_max_blurred" tells the runtime the graph's
# anomaly_score output is NOT the number to threshold on for this architecture
# (unlike SuperSimpleNet's dedicated classification head) - eval.py's calibrated
# threshold is computed on max(cv2.GaussianBlur(map, (15,15), sigma=0)), and the
# graph deliberately omits that blur (see module docstring), so the runtime must
# reproduce it host-side before scoring. Getting this wrong silently produces a
# score the ONNX graph did emit, just not the one any threshold was ever
# calibrated against - a much harder bug to notice than a crash.
EXPORT_METADATA = {
    "anomaly_export_contract": "1.0",
    "architecture": "sk_rd4ad",
    "score_source": "map_max_blurred",
    "blur_kernel_size": "15",
    "blur_sigma": "0.0",   # 0.0 => let cv2.GaussianBlur derive sigma from kernel_size (matches eval.py)
    "verified": "true",    # confirmed against eval.py's compute_image_anomaly_score_and_map
}


def _write_metadata(onnx_path: Path) -> None:
    import onnx
    m = onnx.load(str(onnx_path))
    for k, v in EXPORT_METADATA.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, str(onnx_path))


def export_fp32(model: nn.Module, onnx_path: Path, device: str):
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
    _write_metadata(onnx_path)


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
                   help="Export with RANDOM weights (ignores checkpoint) and run parity.")
    p.add_argument("--no_verify", action="store_true",
                   help="Skip the PyTorch<->ONNX parity check after a real export.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.self_test:
        # With --self_test there is no checkpoint to speak of, so a single
        # positional argument means "output path", not "checkpoint" (argparse
        # otherwise binds it to the first positional declared, silently dropping
        # it since self_test never reads args.checkpoint anyway).
        if args.checkpoint is not None and args.output == "sk_rd4ad_selftest.onnx":
            args.output = args.checkpoint
        elif args.checkpoint is not None:
            print("[!] --self_test uses RANDOM weights; the given checkpoint is ignored.")
        ckpt = None
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            sys.exit("ERROR: provide a checkpoint path, or pass --self_test for a random-weight export.")
        if not Path(ckpt).is_file():
            sys.exit(f"Checkpoint not found: {ckpt}")

    model = build_model(ckpt, device)
    out_path = resolve_output_path(args.output, ckpt)

    print(f"\n--- Exporting SK-RD4AD (pure graph) -> {out_path} ---")
    export_fp32(model, out_path, device)
    print(f"[OK] fp32 export: {out_path}")

    # Verify for BOTH self-test and real exports (parity holds regardless of the
    # weights); only skip if the user explicitly opts out.
    if args.self_test or not args.no_verify:
        verify(model, out_path, device)


if __name__ == "__main__":
    main()
