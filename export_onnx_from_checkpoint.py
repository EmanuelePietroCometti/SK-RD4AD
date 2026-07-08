"""
export_onnx.py — Exports an SK-RD4AD checkpoint (.pth) into a single standalone
.onnx file (embedded weights, no external .onnx.data file), ready for
execution in ONNX Runtime (C++/Python) without dependencies on this repo.

It must be executed from the root of the repo (it needs to be able to do
`import model.resnet` etc.).

Usage:
    python export_onnx.py checkpoint.pth output.onnx
    python export_onnx.py checkpoint.pth output.onnx --net wide_res50 --img-size 256

The exported ONNX graph includes:
  - ImageNet normalization (mean/std) applied internally
  - encoder (pretrained ImageNet, frozen) + bottleneck (bn) + decoder
  - anomaly map calculation (multi-scale cosine distance, as in eval.py)
  - 15x15 Gaussian blur (same logic as cv2.GaussianBlur(..., 0))
  - image score = max of the anomaly map

Input:  "input"       float32 (N, 3, H, W), values in [0, 1]
Output: "anomaly_map" float32 (N, 1, H, W)
        "score"       float32 (N, 1)
"""

import argparse
import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_resnet50, de_wide_resnet50_2
import onnx

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# encoder builder -> returns (encoder, bn); decoder builder -> returns decoder
NET_BUILDERS = {
    "wide_res50": (wide_resnet50_2, de_wide_resnet50_2),
    "res18": (resnet18, de_resnet18),
    "res34": (resnet34, de_resnet34),
    "res50": (resnet50, de_resnet50),
}


def make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


class RD4ADStandalone(nn.Module):
    """
    Wraps encoder + bottleneck (bn) + decoder + post-processing into a single
    forward pass, making the exported ONNX self-contained (image in -> map +
    score out), without needing to replicate the eval.py logic on the client side.
    """

    def __init__(self, encoder, bn, decoder, res: int, img_size: int,
                 normalize: bool = True, blur: bool = True, blur_kernel: int = 15):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        self.res = res
        self.img_size = img_size
        self.do_normalize = normalize
        self.do_blur = blur

        if normalize:
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if blur:
            # same formula used by OpenCV to derive sigma when sigma=0
            sigma = 0.3 * ((blur_kernel - 1) * 0.5 - 1) + 0.8
            self.register_buffer("blur_kernel", make_gaussian_kernel(blur_kernel, sigma))
            self.blur_pad = blur_kernel // 2

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        if self.do_normalize:
            x = (x - self.mean) / self.std

        feats = self.encoder(x)                       # [feature_a, feature_b, feature_c]
        embedding = self.bn(feats)
        recon = self.decoder(embedding, feats[0:3], self.res)

        anomaly_map = None
        for a, b in zip(feats, recon):
            a_n = F.normalize(a, p=2, dim=1)
            b_n = F.normalize(b, p=2, dim=1)
            dist = 1.0 - torch.sum(a_n * b_n, dim=1, keepdim=True)
            dist = F.interpolate(dist, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
            anomaly_map = dist if anomaly_map is None else anomaly_map + dist

        if self.do_blur:
            padded = F.pad(anomaly_map, [self.blur_pad] * 4, mode="reflect")
            anomaly_map = F.conv2d(padded, self.blur_kernel)

        score = torch.amax(anomaly_map, dim=(2, 3))    # (N, 1)
        return anomaly_map, score


def build_model(net: str, res: int, img_size: int, normalize: bool, blur: bool,
                blur_kernel: int, checkpoint_path: str) -> RD4ADStandalone:
    if net not in NET_BUILDERS:
        raise ValueError(f"Unsupported network: '{net}'. Choose from {list(NET_BUILDERS)}")
    enc_fn, dec_fn = NET_BUILDERS[net]

    print(f"[1/3] Building encoder '{net}' with ImageNet pretrained weights (torchvision)...")
    encoder, bn = enc_fn(pretrained=True)
    decoder = dec_fn(pretrained=False)

    print(f"[2/3] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
        if "bn" in checkpoint:
            bn.load_state_dict(checkpoint["bn"])
        else:
            print("      WARNING: checkpoint does not contain 'bn' weights "
                  "-> using random initialization for the bottleneck.")
    else:
        decoder.load_state_dict(checkpoint)
        print("      WARNING: checkpoint contained only the decoder "
              "-> using random initialization for the bottleneck ('bn').")

    encoder.eval()
    bn.eval()
    decoder.eval()

    model = RD4ADStandalone(encoder, bn, decoder, res=res, img_size=img_size,
                            normalize=normalize, blur=blur, blur_kernel=blur_kernel)
    model.eval()
    return model


def export(args: argparse.Namespace) -> None:
    model = build_model(args.net, args.res, args.img_size, not args.no_normalize,
                        not args.no_blur, args.blur_kernel, args.checkpoint)

    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "anomaly_map": {0: "batch"},
            "score": {0: "batch"},
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["anomaly_map", "score"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    # on torch >= 2.5 the default export can switch to the "dynamo" exporter;
    # for this graph (branching on python int, no data-dependent control flow)
    # the classic exporter is more reliable, so we force it if available.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    print(f"[3/3] Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(out_path), **export_kwargs)

    external = out_path.with_name(out_path.name + ".data")
    if external.exists():
        print(f"      WARNING: an external file '{external.name}' was also generated "
              "(the model exceeds the 2GB protobuf limit for a single file).")
    else:
        print("      No external .onnx.data file: standalone model in a single file.")

    try:
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        print("      ONNX structure verification: OK.")
    except ImportError:
        print("      ('onnx' package not installed: skipping structural verification)")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. File: {out_path.resolve()}  ({size_mb:.1f} MB)")
    print(f"Input  'input'       : float32 (N, 3, {args.img_size}, {args.img_size}), "
          + ("values in [0,1] (ImageNet normalization applied inside the graph)"
             if not args.no_normalize else
             "ALREADY ImageNet normalized (mean/std) — normalization excluded from graph"))
    print(f"Output 'anomaly_map' : float32 (N, 1, {args.img_size}, {args.img_size})")
    print(f"Output 'score'       : float32 (N, 1)  — max of the anomaly map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=str, help="Path to the .pth checkpoint (contains 'bn' + 'decoder')")
    p.add_argument("output", type=str, help="Path of the .onnx file to generate")
    p.add_argument("--net", type=str, default="wide_res50", choices=list(NET_BUILDERS),
                   help="Backbone used during training (default: wide_res50)")
    p.add_argument("--res", type=int, default=3,
                   help="Skip-connection parameter used during training: 0 = no skip, "
                        "any value != 0 = skips active (default: 3, the project default)")
    p.add_argument("--img-size", type=int, default=256, help="Image side, H=W (default: 256)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    p.add_argument("--fixed-batch", action="store_true",
                   help="Fix batch=1 in the graph instead of keeping it dynamic")
    p.add_argument("--no-normalize", action="store_true",
                   help="Do not include ImageNet normalization in the graph "
                        "(use this if you already normalize on the C++ side)")
    p.add_argument("--no-blur", action="store_true",
                   help="Do not include the 15x15 Gaussian blur in the graph")
    p.add_argument("--blur-kernel", type=int, default=15,
                   help="Gaussian blur kernel size (must be odd) (default: 15)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.checkpoint).is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    export(args)