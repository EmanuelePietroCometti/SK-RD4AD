"""
calibrate_threshold.py — Compute the decision threshold THROUGH the ONNX model.

Why this exists
---------------
A threshold is only valid for the exact pipeline that produced the scores it
was computed on. This script therefore runs the PRODUCTION pipeline end to
end — canonical preprocessing (resize 256 -> [0,1] -> dynamic crop ->
ImageNet normalize, identical to main.py's training loop) followed by the
exported ONNX graph (contract 2.0: blur and score baked in) — over a labeled
MVTec-style test set, and derives:

    - the F1-optimal image-level threshold (same criterion as eval.py),
    - global_min / global_max of the blurred maps (for eval.py-style
      threshold-centric heatmap visualization).

Results are written to a sidecar JSON next to the model and, with --embed,
into the .onnx metadata under the keys the inference runtime reads
("calibrated_threshold", "calibration_global_min", "calibration_global_max").

Because eval.py now applies the same canonical pipeline in PyTorch, its
calibration_pytorch.json must agree with this one up to floating-point drift;
a larger gap means the preprocessing has diverged — investigate before
shipping.

Usage
-----
    python calibrate_threshold.py --model out.onnx --data_path ./mvtec/ \
        --class_ bottle [--seg 1] [--embed]
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)

from dataset.dataset import get_data_transforms, MVTecDataset, MVTecDataset_no_seg
from test import apply_dynamic_crop_gpu

IMG_SIZE = 256


def load_session(model_path):
    import onnx
    import onnxruntime as ort

    meta = {p.key: p.value for p in onnx.load(model_path).metadata_props}
    score_source = meta.get("score_source")
    if score_source != "graph":
        sys.exit(
            f"ERROR: model has score_source={score_source!r}, expected 'graph' "
            "(anomaly_export_contract 2.0, blur baked into the graph).\n"
            "Re-export the checkpoint with the current "
            "export_onnx_from_checkpoint.py: a threshold calibrated here would "
            "not apply to a contract-1.0 model (its score is un-blurred)."
        )
    if meta.get("weights_source") == "random_self_test":
        sys.exit("ERROR: refusing to calibrate a --self_test model (random weights).")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers), meta


def canonical_preprocess(img):
    """Dataset tensors arrive already resized+normalized; the dynamic crop must
    run on the [0,1] image (main.py convention): denormalize -> crop -> renorm."""
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img * std_t + mean_t).clamp(0, 1)
    img = apply_dynamic_crop_gpu(img)
    return (img - mean_t) / std_t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to the .onnx model (contract 2.0)")
    p.add_argument("--data_path", default="./mvtec/", help="Dataset root")
    p.add_argument("--class_", required=True, help="Dataset class (subfolder of data_path)")
    p.add_argument("--seg", default=1, type=int, choices=[0, 1],
                   help="1: MVTecDataset (with masks), 0: MVTecDataset_no_seg")
    p.add_argument("--embed", action="store_true",
                   help="Also write the calibration into the .onnx metadata")
    p.add_argument("--output", default=None,
                   help="Calibration JSON path (default: <model>.calibration.json)")
    args = p.parse_args()

    sess, meta = load_session(args.model)

    data_transform, gt_transform = get_data_transforms(IMG_SIZE, IMG_SIZE)
    test_path = os.path.join(args.data_path, args.class_)
    if args.seg == 1:
        dataset = MVTecDataset(root=test_path, transform=data_transform,
                               gt_transform=gt_transform, phase="test")
    else:
        dataset = MVTecDataset_no_seg(root=test_path, transform=data_transform,
                                      phase="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    y_true, scores, map_mins = [], [], []
    print(f"Scoring {len(dataset)} images through the ONNX pipeline...")
    for batch in loader:
        img = batch[0]
        label = batch[2] if args.seg == 1 else batch[1]
        img = canonical_preprocess(img)

        amap, score = sess.run(["anomaly_map", "anomaly_score"],
                               {"input_tensor": img.numpy().astype(np.float32)})
        y_true.append(int(label.item()))
        scores.append(float(score[0]))
        map_mins.append(float(amap.min()))

    y_true = np.array(y_true)
    scores = np.array(scores)
    if len(np.unique(y_true)) < 2:
        sys.exit("ERROR: the test set must contain both good and defective images "
                 "to calibrate an F1-optimal threshold.")

    # Same criterion as eval.py: F1-optimal threshold on the PR curve.
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = min(np.argmax(f1_scores), len(thresholds) - 1)
    threshold = float(thresholds[best_idx])

    y_pred = (scores >= threshold).astype(int)
    report = {
        "pipeline": "dynamic_crop -> sum(1-cos, align_corners=False) -> "
                    "gauss_k15_sigma4_zeropad -> max  [blur+score in-graph]",
        "source": "calibrate_threshold.py (ONNX production pipeline)",
        "model": os.path.basename(args.model),
        "class": args.class_,
        "date": datetime.date.today().isoformat(),
        "n_images": int(len(y_true)),
        "threshold": threshold,
        "global_min": float(min(map_mins)),
        "global_max": float(scores.max()),
        "auroc_sp": float(roc_auc_score(y_true, scores)),
        "f1_sp": float(f1_score(y_true, y_pred)),
        "precision_sp": float(precision_score(y_true, y_pred)),
        "recall_sp": float(recall_score(y_true, y_pred)),
        "accuracy_sp": float(accuracy_score(y_true, y_pred)),
    }

    print("=" * 50)
    print(" ONNX CALIBRATION REPORT ")
    print("=" * 50)
    for k in ("threshold", "auroc_sp", "f1_sp", "precision_sp", "recall_sp", "accuracy_sp"):
        print(f"{k:15s}: {report[k]:.4f}")
    print("Confusion Matrix (TN, FP | FN, TP):")
    print(confusion_matrix(y_true, y_pred))
    print("=" * 50)

    out_path = args.output or (os.path.splitext(args.model)[0] + ".calibration.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Calibration saved to {out_path}")

    if args.embed:
        import onnx
        m = onnx.load(args.model)
        embed = {
            "calibrated_threshold": repr(threshold),
            "calibration_global_min": repr(report["global_min"]),
            "calibration_global_max": repr(report["global_max"]),
            "calibration_date": report["date"],
            "calibration_class": args.class_,
            # Shown by the inference runtime next to the threshold it loads.
            "calibration_info": (f"F1-optimal on labeled '{args.class_}' test set "
                                 f"({len(y_true)} imgs, ONNX pipeline), {report['date']}"),
        }
        existing = {p_.key: p_ for p_ in m.metadata_props}
        for k, v in embed.items():
            if k in existing:
                existing[k].value = v
            else:
                entry = m.metadata_props.add()
                entry.key, entry.value = k, v
        onnx.save(m, args.model)
        print(f"Calibration embedded into {args.model} metadata.")


if __name__ == "__main__":
    main()
