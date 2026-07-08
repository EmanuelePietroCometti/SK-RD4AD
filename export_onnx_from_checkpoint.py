"""
export_onnx_from_checkpoint.py — Exports an SK-RD4AD checkpoint (.pth) into a
single standalone .onnx file ready for ONNX Runtime in production.

All architecture/preprocessing parameters are FIXED to the training values
(main.py / dataset.py / eval.py): wide_res50 backbone, res=3 skip connections,
256x256 input, ImageNet normalization, 15x15 Gaussian blur post-processing.
The only configurable graph parameter is the batch size; precision can be
fp32, fp16 or int8.

The exported graph is fully self-contained (raw image in -> anomaly map +
score out). No preprocessing or postprocessing is needed on the client:

  Input : "input"       uint8   (N, 3, H, W)  RGB, values 0..255, any H/W
          (the graph internally converts to float, resizes to 256x256 with
           bilinear+antialias — same as v2.Resize used at train/eval time —
           and applies ImageNet mean/std normalization)
  Output: "anomaly_map" float32 (N, 1, 256, 256)
          "score"       float32 (N, 1)   — max of the blurred anomaly map

Usage (run from the repo root):
    python export_onnx_from_checkpoint.py checkpoint.pth output.onnx
    python export_onnx_from_checkpoint.py checkpoint.pth output.onnx --batch-size 8
    python export_onnx_from_checkpoint.py checkpoint.pth output.onnx --batch-size dynamic
    python export_onnx_from_checkpoint.py checkpoint.pth output.onnx --precision fp16
    python export_onnx_from_checkpoint.py checkpoint.pth output.onnx --precision int8 --calib-dir mvtec/bottle/train/good
"""

import argparse
import copy
import glob
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2

# ---------------------------------------------------------------------------
# Fixed training-time parameters (do not change: they must match main.py)
# ---------------------------------------------------------------------------
NET = "wide_res50"      # main.py --net default
RES = 3                 # main.py --res default (skip connections active)
IMG_SIZE = 256          # main.py: image_size = 256
BLUR_KERNEL = 15        # eval.py: cv2.GaussianBlur(..., (15, 15), 0)
OPSET = 18              # needed for Resize with antialias (matches v2.Resize)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    # same sigma formula used by OpenCV when sigma=0
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


class RD4ADStandalone(nn.Module):
    """
    encoder + bottleneck + decoder + full pre/post-processing in one forward,
    so the exported ONNX is raw-image-in -> map+score-out.
    """

    def __init__(self, encoder, bn, decoder):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        self.fp16 = False
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("blur_kernel", make_gaussian_kernel(BLUR_KERNEL))
        self.blur_pad = BLUR_KERNEL // 2
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # --- preprocessing (same as dataset.get_data_transforms) ---
        x = x.to(torch.float32) / 255.0                      # ToDtype(scale=True)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE),      # v2.Resize(antialias=True)
                          mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std                       # v2.Normalize

        # --- model (fp16 export runs the networks in half precision, while
        # pre/post-processing stays fp32 for numerical stability) ---
        if self.fp16:
            x = x.to(torch.float16)
        feats = self.encoder(x)
        recon = self.decoder(self.bn(feats), feats[0:3], RES)

        # --- postprocessing (same as eval.py) ---
        anomaly_map = None
        for a, b in zip(feats, recon):
            a_n = F.normalize(a.to(torch.float32), p=2, dim=1)
            b_n = F.normalize(b.to(torch.float32), p=2, dim=1)
            dist = 1.0 - torch.sum(a_n * b_n, dim=1, keepdim=True)
            dist = F.interpolate(dist, size=(IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            anomaly_map = dist if anomaly_map is None else anomaly_map + dist

        padded = F.pad(anomaly_map, [self.blur_pad] * 4, mode="reflect")
        anomaly_map = F.conv2d(padded, self.blur_kernel)
        score = torch.amax(anomaly_map, dim=(2, 3))
        return anomaly_map, score

    def to_fp16(self) -> "RD4ADStandalone":
        self.encoder.half()
        self.bn.half()
        self.decoder.half()
        self.fp16 = True
        return self


def build_model(checkpoint_path: str) -> RD4ADStandalone:
    print(f"[1/4] Building '{NET}' encoder (ImageNet pretrained, frozen at training time)...")
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    print(f"[2/4] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
        if "bn" in checkpoint:
            bn.load_state_dict(checkpoint["bn"])
        else:
            sys.exit("ERROR: checkpoint does not contain 'bn' weights; an export "
                     "with a random bottleneck would be useless in production.")
    else:
        sys.exit("ERROR: checkpoint must be a dict with 'bn' and 'decoder' keys "
                 "(as saved by main.py).")

    model = RD4ADStandalone(encoder.eval(), bn.eval(), decoder.eval())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Precision conversions
# ---------------------------------------------------------------------------
class _RandomCalibReader:
    """Fallback calibration with random data — functional but inaccurate."""

    def __init__(self, batch_size: int, n_batches: int = 8):
        self.data = iter(
            {"input": np.random.randint(0, 256, (batch_size, 3, IMG_SIZE, IMG_SIZE),
                                        dtype=np.uint8)}
            for _ in range(n_batches)
        )

    def get_next(self):
        return next(self.data, None)


class _ImageCalibReader:
    """Calibration from a folder of training images (recommended)."""

    def __init__(self, calib_dir: str, batch_size: int, max_images: int = 64):
        import cv2
        exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.bmp")
        paths = sorted(p for e in exts for p in glob.glob(str(Path(calib_dir) / "**" / e), recursive=True))
        if not paths:
            sys.exit(f"ERROR: no images found in calibration dir: {calib_dir}")
        paths = paths[:max_images]
        print(f"      Calibrating on {len(paths)} images from {calib_dir}")
        batches = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            if len(chunk) < batch_size:
                break
            imgs = []
            for p in chunk:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                imgs.append(img.transpose(2, 0, 1))
            batches.append({"input": np.stack(imgs).astype(np.uint8)})
        self.data = iter(batches)

    def get_next(self):
        return next(self.data, None)


def convert_int8(fp32_path: Path, out_path: Path, batch_size: int, calib_dir: str | None) -> None:
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if calib_dir:
        reader = _ImageCalibReader(calib_dir, batch_size)
    else:
        print("      WARNING: no --calib-dir given: calibrating on RANDOM data. "
              "Accuracy will suffer; pass a folder of 'good' training images.")
        reader = _RandomCalibReader(batch_size)

    print("      Static int8 quantization (QDQ, Conv only, per-channel)...")
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        # symbolic shape inference chokes on the dynamic H/W input; standard
        # shape inference + optimization is enough for QDQ quantization
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        quantize_static(
            str(pre), str(out_path), reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "ConvTranspose"],
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(onnx_path: Path, torch_model: RD4ADStandalone, batch_size: int, precision: str) -> None:
    import onnxruntime as ort
    print(f"[4/4] Verifying ONNX vs PyTorch (batch={batch_size})...")
    onnx.checker.check_model(str(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # non-square input to also exercise the fused resize
    x = rng.integers(0, 256, (batch_size, 3, 320, 288), dtype=np.uint8)

    ort_map, ort_score = sess.run(None, {"input": x})
    with torch.no_grad():
        ref_map, ref_score = torch_model(torch.from_numpy(x))

    map_diff = np.abs(ort_map - ref_map.numpy()).max()
    score_diff = np.abs(ort_score - ref_score.numpy()).max()
    tol = {"fp32": 1e-4, "fp16": 5e-3, "int8": float("inf")}[precision]
    status = "OK" if max(map_diff, score_diff) <= tol else "WARNING: above expected tolerance"
    print(f"      max |Δ anomaly_map| = {map_diff:.6f}, max |Δ score| = {score_diff:.6f} "
          f"(tolerance {precision}: {tol}) -> {status}")
    if precision == "int8":
        print("      (int8 has no strict tolerance: validate AUROC on your test set)")


# ---------------------------------------------------------------------------
def export(args: argparse.Namespace) -> None:
    model = build_model(args.checkpoint)

    dynamic = args.batch_size == "dynamic"
    batch = 1 if dynamic else int(args.batch_size)
    if not dynamic and batch < 1:
        sys.exit("ERROR: --batch-size must be >= 1 or 'dynamic'")

    dummy = torch.randint(0, 256, (batch, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    dynamic_axes = {"input": {2: "height", 3: "width"},
                    "anomaly_map": {}, "score": {}}
    if dynamic:
        dynamic_axes["input"][0] = "batch"
        dynamic_axes["anomaly_map"][0] = "batch"
        dynamic_axes["score"][0] = "batch"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["anomaly_map", "score"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        # the dynamo exporter is required: the classic one cannot export the
        # antialiased bilinear resize (aten::_upsample_bilinear2d_aa) that
        # replicates v2.Resize(antialias=True) used at train/eval time
        dynamo=True,
        external_data=False,
        optimize=True,
    )

    print(f"[3/4] Exporting ONNX (opset {OPSET}, precision {args.precision}, "
          f"batch {'dynamic' if dynamic else batch}) -> {out_path}")
    export_model = copy.deepcopy(model).to_fp16() if args.precision == "fp16" else model
    with tempfile.TemporaryDirectory() as tmp:
        direct_path = out_path if args.precision != "int8" else Path(tmp) / "fp32.onnx"
        with torch.no_grad():
            torch.onnx.export(export_model, dummy, str(direct_path), **export_kwargs)
        if args.precision == "int8":
            convert_int8(direct_path, out_path, batch, args.calib_dir)

    # always verify against the fp32 PyTorch reference (training numerics)
    verify(out_path, model, batch, args.precision)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. File: {out_path.resolve()}  ({size_mb:.1f} MB)")
    print(f"Input  'input'       : uint8 (N, 3, H, W) RGB 0..255 — resize + "
          f"normalization are inside the graph")
    print(f"Output 'anomaly_map' : float32 (N, 1, {IMG_SIZE}, {IMG_SIZE})")
    print(f"Output 'score'       : float32 (N, 1) — max of the blurred anomaly map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=str, help="Path to the .pth checkpoint ('bn' + 'decoder')")
    p.add_argument("output", type=str, help="Path of the .onnx file to generate")
    p.add_argument("--batch-size", type=str, default="1",
                   help="Batch size baked into the graph, or 'dynamic' (default: 1)")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32",
                   help="Export precision (default: fp32)")
    p.add_argument("--calib-dir", type=str, default=None,
                   help="Folder of 'good' training images for int8 calibration "
                        "(strongly recommended with --precision int8)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.checkpoint).is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    export(args)
