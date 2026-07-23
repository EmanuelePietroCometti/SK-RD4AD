"""
parity_check.py — Verify that the PyTorch reference pipeline and the exported
ONNX graph produce the same anomaly maps and scores on REAL images.

Both paths share the same host preprocessing up to the [0,1] dynamic-cropped
image (resize -> [0,1] -> dynamic crop). Under contract 3.0 normalization is
in-graph, so the ONNX side is fed the [0,1] image and normalizes internally,
while the PyTorch reference is fed the normalized tensor and applies test.py's
canonical map definition (compute_anomaly_map_torch); the graph bakes the same
definition (normalize + cosine map + blur + score) in. Any gap beyond
floating-point kernel differences (~1e-4) means the two pipelines have drifted —
do not ship the model until this passes.

Bit-exact equality is impossible (PyTorch and onnxruntime use different conv
kernels); the pass criterion is max |Δ| < 1e-3 on both map and score, in fp32.

Usage
-----
    python parity_check.py --checkpoint ckpt.pth --model out.onnx \
        --data_path ./mvtec/ --class_ bottle [--seg 1] [--res 3] [--n 20]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import get_data_transforms, MVTecDataset, MVTecDataset_no_seg
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from test import apply_dynamic_crop_gpu, compute_anomaly_map_torch

IMG_SIZE = 256
TOLERANCE = 1e-3


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint")
    p.add_argument("--model", required=True, help="Path to the exported .onnx")
    p.add_argument("--data_path", default="./mvtec/", help="Dataset root")
    p.add_argument("--class_", required=True, help="Dataset class")
    p.add_argument("--seg", default=1, type=int, choices=[0, 1])
    p.add_argument("--res", default=3, type=int, choices=[1, 2, 3],
                   help="MUST match the --res used at training AND export time")
    p.add_argument("--n", default=20, type=int, help="Number of images to compare")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- PyTorch reference (same construction as eval.py) ---
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(ckpt["decoder"])
    bn.load_state_dict(ckpt["bn"])
    encoder, bn, decoder = encoder.to(device).eval(), bn.to(device).eval(), decoder.to(device).eval()

    # --- ONNX session ---
    import onnx
    import onnxruntime as ort
    meta = {m.key: m.value for m in onnx.load(args.model).metadata_props}
    if meta.get("export_contract") != "3.0":
        sys.exit("ERROR: model is not contract 3.0 (export_contract != '3.0'); "
                 "re-export with the current export_onnx_from_checkpoint.py.")
    if str(meta.get("res", args.res)) != str(args.res):
        sys.exit(f"ERROR: model was exported with res={meta.get('res')} but "
                 f"--res {args.res} was requested; the comparison would be meaningless.")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)

    # --- Data ---
    data_transform, gt_transform = get_data_transforms(IMG_SIZE, IMG_SIZE)
    test_path = os.path.join(args.data_path, args.class_)
    if args.seg == 1:
        dataset = MVTecDataset(root=test_path, transform=data_transform,
                               gt_transform=gt_transform, phase="test")
    else:
        dataset = MVTecDataset_no_seg(root=test_path, transform=data_transform,
                                      phase="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    worst_map, worst_score = 0.0, 0.0
    n_done = 0
    print(f"Comparing PyTorch vs ONNX on up to {args.n} images (tolerance {TOLERANCE:g})...")
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.n:
                break
            img = batch[0].to(device)

            # Canonical host preprocessing: denormalize -> [0,1] -> dynamic crop.
            # Under contract 3.0 the ONNX graph receives this [0,1] image and
            # normalizes internally (InGraphNormalize); the PyTorch reference
            # encoder expects the normalized tensor, so it is normalized here.
            img01 = (img * std_t + mean_t).clamp(0, 1)
            img01 = apply_dynamic_crop_gpu(img01)
            img_norm = (img01 - mean_t) / std_t

            # PyTorch path (normalized input)
            inputs = encoder(img_norm)
            outputs = decoder(bn(inputs), inputs[0:3], args.res)
            t_map, _ = compute_anomaly_map_torch(inputs[0:3], outputs, IMG_SIZE)
            t_map = t_map.cpu().numpy()
            t_score = float(t_map.max())

            # ONNX path (contract 3.0: [0,1] input, normalization in-graph)
            o_map, o_score = sess.run(["anomaly_map", "anomaly_score"],
                                      {"image": img01.cpu().numpy().astype(np.float32)})

            d_map = float(np.abs(o_map - t_map).max())
            d_score = float(abs(float(o_score[0]) - t_score))
            worst_map, worst_score = max(worst_map, d_map), max(worst_score, d_score)
            n_done += 1
            print(f"  img {n_done:3d}: score torch={t_score:.6f} onnx={float(o_score[0]):.6f} "
                  f"|Δscore|={d_score:.2e}  |Δmap|max={d_map:.2e}")

    print("-" * 60)
    print(f"Worst over {n_done} images: |Δscore|={worst_score:.2e}  |Δmap|max={worst_map:.2e}")
    if worst_map < TOLERANCE and worst_score < TOLERANCE:
        print(f"[PASS] PyTorch and ONNX pipelines agree within {TOLERANCE:g}.")
    else:
        sys.exit(f"[FAIL] Divergence exceeds {TOLERANCE:g}: the pipelines have drifted. "
                 "Check res, checkpoint/onnx pairing, and preprocessing.")


if __name__ == "__main__":
    main()
