import torch
from dataset.dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset.dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, f1_score, 
                             precision_score, recall_score, accuracy_score)
import matplotlib
import pickle
import os
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import normalize
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F_v2
import shutil

from eval_config import EvalConfig

plt.switch_backend('agg')

def apply_dynamic_crop_gpu(images, masks=None, padding=30):
    """
    Dynamic crop to center region of interest, executed entirely on GPU.
    Supports optional mask resizing for test/evaluation phase.
    """
    B, C, H, W = images.shape
    cropped_imgs = []
    cropped_masks = [] if masks is not None else None
    
    gray = images.mean(dim=1)
    is_dark = gray < 0.94
    
    for i in range(B):
        coords = torch.nonzero(is_dark[i])
        if coords.numel() == 0:
            cropped_imgs.append(images[i])
            if masks is not None: cropped_masks.append(masks[i])
            continue
            
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values
        size = torch.maximum(y_max - y_min, x_max - x_min)
        cy, cx = y_min + (y_max - y_min) // 2, x_min + (x_max - x_min) // 2
        
        y1, y2 = torch.clamp(cy - size//2 - padding, min=0), torch.clamp(cy + size//2 + padding, max=H)
        x1, x2 = torch.clamp(cx - size//2 - padding, min=0), torch.clamp(cx + size//2 + padding, max=W)
        
        crop_img = images[i:i+1, :, y1:y2, x1:x2]
        # InterpolationMode.BILINEAR = 2
        cropped_imgs.append(F_v2.resize(crop_img, size=[H, W], interpolation=2, antialias=True).squeeze(0))
        
        if masks is not None:
            # Handle mask dims correctly (can be 3D or 4D depending on dataset)
            crop_mask = masks[i:i+1, y1:y2, x1:x2] if masks.dim() == 3 else masks[i:i+1, :, y1:y2, x1:x2]
            # InterpolationMode.NEAREST = 0
            resized_mask = F_v2.resize(crop_mask, size=[H, W], interpolation=0)
            cropped_masks.append(resized_mask.squeeze(0))
            
    if masks is not None:
        return torch.stack(cropped_imgs), torch.stack(cropped_masks)
    return torch.stack(cropped_imgs)

# ---------------------------------------------------------------------------
# ANOMALY-MAP DEFINITION — single source of truth for the whole repo.
#
# Map = sum over the three layers of (1 - cosine similarity), bilinearly
# upsampled, then Gaussian-blurred; image-level score = max of the BLURRED map.
# Every free choice (blur sigma / kernel size / border handling, align_corners,
# dynamic crop at evaluation) lives in EvalConfig (eval_config.py):
#
#   preset "paper"     (default) the authors' repo: scipy gaussian_filter(sigma=4)
#                      -> 33x33 kernel, symmetric ('reflect') borders,
#                      align_corners=True, no crop at evaluation.
#   preset "canonical" the former definition of this fork, ONNX-friendly:
#                      15x15 kernel, zero padding, align_corners=False.
#
# GAUSS_KERNEL_SIZE / GAUSS_SIGMA / get_gaussian_kernel below describe the
# "canonical" preset and are kept unchanged: an ONNX exporter that imports them
# keeps baking the 15x15 zero-padded blur. Thresholds in calib.json are valid
# only for the score they were computed with (see "score_definition" there).
# ---------------------------------------------------------------------------
GAUSS_KERNEL_SIZE = 15
GAUSS_SIGMA = 4.0

_DEFAULT_EVAL_CFG = EvalConfig()  # preset "paper", crop off


def _resolve_eval_cfg(eval_cfg):
    return _DEFAULT_EVAL_CFG.resolved() if eval_cfg is None else eval_cfg


def gaussian_kernel_2d(size, sigma, device):
    """Normalised 2-D Gaussian kernel [1, 1, size, size] (outer product of the 1-D profile)."""
    x = torch.arange(size, device=device).float() - size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    kernel = gauss[:, None] * gauss[None, :]
    return (kernel / kernel.sum()).view(1, 1, size, size)


def get_gaussian_kernel(device):
    # preset "canonical" kernel (bit-identical to the previous implementation)
    return gaussian_kernel_2d(GAUSS_KERNEL_SIZE, GAUSS_SIGMA, device)


def _symmetric_index(n, p, device):
    """Indices of the half-sample symmetric extension (d c b a | a b c d | d c b a) of a length-n axis.

    This is scipy's mode='reflect' (default of gaussian_filter). torch's
    padding_mode='reflect' is whole-sample and therefore NOT equivalent.
    """
    i = torch.arange(-p, n + p, device=device) % (2 * n)
    return torch.where(i >= n, 2 * n - 1 - i, i)


def blur_anomaly_map(anomaly_map, eval_cfg=None):
    """Gaussian blur of a [B, 1, H, W] map according to EvalConfig."""
    cfg = _resolve_eval_cfg(eval_cfg)
    size = cfg.kernel_size()
    kernel = gaussian_kernel_2d(size, cfg.blur_sigma, anomaly_map.device)
    pad = size // 2
    if cfg.blur_padding == 'zeros':
        return F.conv2d(anomaly_map, kernel, padding=pad)
    # 'symmetric': pad explicitly, then valid convolution
    h, w = anomaly_map.shape[-2:]
    idx_h = _symmetric_index(h, pad, anomaly_map.device)
    idx_w = _symmetric_index(w, pad, anomaly_map.device)
    padded = anomaly_map.index_select(2, idx_h).index_select(3, idx_w)
    return F.conv2d(padded, kernel, padding=0)


def compute_anomaly_map_torch(fs_list, ft_list, out_size, eval_cfg=None):
    """Blurred anomaly map as a tensor [B, 1, out_size, out_size].

    Returns (blurred_map, per_layer_unblurred_maps). Image-level score is
    blurred_map.amax() over the spatial dims. eval_cfg=None => preset "paper".
    """
    cfg = _resolve_eval_cfg(eval_cfg)
    anomaly_map = None
    layer_maps = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1).unsqueeze(1)  # (B,1,h,w)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=cfg.align_corners)
        layer_maps.append(a_map)
        anomaly_map = a_map if anomaly_map is None else anomaly_map + a_map

    blurred = blur_anomaly_map(anomaly_map, cfg)
    return blurred, layer_maps

# Calculate anomaly score map (numpy wrapper around the definition above)
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='a', eval_cfg=None):
    # amap_mode is kept for signature compatibility; only the additive mode
    # exists now (the 'mul' branch was unused).
    blurred, layer_maps = compute_anomaly_map_torch(fs_list, ft_list, out_size, eval_cfg)
    a_map_list = [m.squeeze().cpu().detach().numpy() for m in layer_maps]
    return blurred.squeeze().cpu().detach().numpy(), a_map_list

# Visualize anomaly map over image
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Normalize the image between 0 and 1
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

# Convert image to heatmap
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

# Compute Per-Region Overlap (PRO) and Area Under the Curve (AUC)
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlapping (PRO) and 0 to 0.3 FPR
    Args:
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"

    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    metrics_records = []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        metrics_records.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame(metrics_records)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

# Evaluation function without segmentation
def evaluation_me(encoder, bn, decoder, res, dataloader, device, print_canshu, score_num, eval_cfg=None):
    decoder.eval()
    bn.eval()
    encoder.eval()
    eval_cfg = _resolve_eval_cfg(eval_cfg)

    # Lists to store sample-level labels and predictions
    gt_list_sp = []
    pr_list_sp = []


    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for (img, label, _, _) in dataloader:
            img = img.to(device)

            # Optional dynamic crop (EvalConfig.dynamic_crop; the paper does not crop).
            # Denormalize -> crop -> renormalize, only when requested.
            if eval_cfg.dynamic_crop:
                img = (img * std_t + mean_t).clamp(0, 1)
                img = apply_dynamic_crop_gpu(img)
                img = (img - mean_t) / std_t

            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)

            # Calculate final anomaly map (supports any batch size)
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a', eval_cfg=eval_cfg)
            anomaly_map = anomaly_map.reshape(img.shape[0], -1)

            # Add sample-level labels
            gt_list_sp.extend(label.numpy().tolist())

            # Sample-level prediction: mean of the top-`score_num` pixels per image
            top_scores = np.sort(anomaly_map, axis=1)[:, -score_num:].mean(axis=1)
            pr_list_sp.extend(np.round(top_scores, 3).tolist())

        if print_canshu == 1:
            print(gt_list_sp, pr_list_sp)  # Print intermediate results

        # Calculate sample-level AUROC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
    
    return auroc_sp

# Generate heatmaps for evaluation visualization (segmentation version).
# Mirrors evaluation_visualization_no_seg: images are bucketed into
# tp/tn/fp/fn subfolders and the confusion matrix is returned to the caller.
def evaluation_visualization(encoder, bn, decoder, res, dataloader, device,
                             print_canshu, score_num, img_path,
                             threshold=None, nest_by_type=True, eval_cfg=None):
    decoder.eval(); bn.eval(); encoder.eval()
    eval_cfg = _resolve_eval_cfg(eval_cfg)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    records = []
    with torch.no_grad():
        for img, gt, label, img_type, ip in dataloader:   # batch_size must be 1
            img = img.to(device)
            gt = gt.to(device)

            # `img` below is the [0,1] image used for the overlay; `img_norm` is the
            # model input. Without crop the model input is the dataloader tensor itself.
            img_norm = img
            img = (img * std_t + mean_t).clamp(0, 1)
            if eval_cfg.dynamic_crop:
                img, gt = apply_dynamic_crop_gpu(img, masks=gt)
                img_norm = (img - mean_t) / std_t

            inputs = encoder(img_norm)
            outputs = decoder(bn(inputs), inputs[0:3], res)

            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a', eval_cfg=eval_cfg)
            gt = (gt > 0.5).float()

            # Image-level score = max of the blurred map: the same definition
            # evaluation() uses, so this confusion matrix is consistent with the
            # F1/precision/recall printed in the report.
            records.append((int(label.item()), float(anomaly_map.max()),
                            img[0].cpu().numpy(), anomaly_map,
                            gt.cpu().numpy().astype(int)[0][0],
                            img_type[0], ip[0]))

    labels = np.array([r[0] for r in records])
    scores = np.array([r[1] for r in records])

    if threshold is None:
        prec, rec, thr = precision_recall_curve(labels, scores)
        f1 = (2 * prec * rec) / (prec + rec + 1e-10)
        threshold = float(thr[min(np.argmax(f1), len(thr) - 1)])

    preds = (scores >= threshold).astype(int)

    metrics = {
        'auroc':     round(roc_auc_score(labels, scores), 4) if len(np.unique(labels)) >= 2 else float('nan'),
        'ap':        round(average_precision_score(labels, scores), 4) if len(np.unique(labels)) >= 2 else float('nan'),
        'f1':        round(f1_score(labels, preds, zero_division=0), 4),
        'precision': round(precision_score(labels, preds, zero_division=0), 4),
        'recall':    round(recall_score(labels, preds, zero_division=0), 4),
        'accuracy':  round(accuracy_score(labels, preds), 4),
        'balanced_accuracy': round((recall_score(labels, preds, zero_division=0) +
                                    recall_score(1 - labels, 1 - preds, zero_division=0)) / 2, 4),
    }

    cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for (lab, sc, rgb01, amap, gt_np, dtype, path), pred in zip(records, preds):
        if   lab == 1 and pred == 1: sub = 'tp'
        elif lab == 0 and pred == 0: sub = 'tn'
        elif lab == 0 and pred == 1: sub = 'fp'
        else:                        sub = 'fn'
        cm[sub] += 1

        out_dir = os.path.join(img_path, sub, dtype) if nest_by_type else os.path.join(img_path, sub)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, stem + '.png')

        heat = cvt2heatmap(255 - min_max_norm(amap) * 255)
        rgb = np.transpose(rgb01, (1, 2, 0)) * 255
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.uint8(min_max_norm(rgb) * 255)
        overlay = show_cam_on_image(rgb, heat)

        fig = plt.figure()
        plt.subplot(1, 3, 1); plt.imshow(overlay);            plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(gt_np * 255, cmap='gray'); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(rgb);                plt.axis('off')
        plt.savefig(out); plt.close(fig)

    return cm, threshold, metrics

# Generate heatmaps for evaluation visualization without segmentation
def evaluation_visualization_no_seg(encoder, bn, decoder, res, dataloader, device,
                                    score_num, img_path,
                                    threshold=None,
                                    save_panel=True, nest_by_type=True, eval_cfg=None):
    decoder.eval(); bn.eval(); encoder.eval()
    eval_cfg = _resolve_eval_cfg(eval_cfg)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    records = []
    with torch.no_grad():
        for img, label, img_type, paths in dataloader:
            img = img.to(device)
            img_norm = img  # model input; replaced only if the dynamic crop is on
            img = (img * std_t + mean_t).clamp(0, 1)
            if eval_cfg.dynamic_crop:
                img = apply_dynamic_crop_gpu(img)
                img_norm = (img - mean_t) / std_t

            inputs = encoder(img_norm)
            outputs = decoder(bn(inputs), inputs[0:3], res)
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a', eval_cfg=eval_cfg)

            flat = anomaly_map.reshape(img.shape[0], -1)
            top_scores = np.sort(flat, axis=1)[:, -score_num:].mean(axis=1)

            for i in range(img.shape[0]):
                records.append((int(label[i].item()), float(top_scores[i]),
                                img[i].cpu().numpy(), anomaly_map[i],
                                img_type[i], paths[i]))

    labels = np.array([r[0] for r in records])
    scores = np.array([r[1] for r in records])

    if threshold is None:
        prec, rec, thr = precision_recall_curve(labels, scores)
        f1 = (2 * prec * rec) / (prec + rec + 1e-10)
        threshold = float(thr[min(np.argmax(f1), len(thr) - 1)])

    preds = (scores >= threshold).astype(int)

    metrics = {
        'auroc':     round(roc_auc_score(labels, scores), 4) if len(np.unique(labels)) >= 2 else float('nan'),
        'ap':        round(average_precision_score(labels, scores), 4) if len(np.unique(labels)) >= 2 else float('nan'),
        'f1':        round(f1_score(labels, preds, zero_division=0), 4),
        'precision': round(precision_score(labels, preds, zero_division=0), 4),
        'recall':    round(recall_score(labels, preds, zero_division=0), 4),
        'accuracy':  round(accuracy_score(labels, preds), 4),
        'balanced_accuracy': round((recall_score(labels, preds, zero_division=0) +
                                    recall_score(1 - labels, 1 - preds, zero_division=0)) / 2, 4),
    }

    cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for (lab, sc, rgb01, amap, dtype, path), pred in zip(records, preds):
        if   lab == 1 and pred == 1: sub = 'tp'
        elif lab == 0 and pred == 0: sub = 'tn'
        elif lab == 0 and pred == 1: sub = 'fp'
        else:                        sub = 'fn'
        cm[sub] += 1

        out_dir = os.path.join(img_path, sub, dtype) if nest_by_type else os.path.join(img_path, sub)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, stem + '.png')

        if save_panel:
            heat = cvt2heatmap(255 - min_max_norm(amap) * 255)
            rgb = np.transpose(rgb01, (1, 2, 0)) * 255
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.uint8(min_max_norm(rgb) * 255)
            overlay = show_cam_on_image(rgb, heat)
            fig = plt.figure()
            plt.subplot(1, 2, 1); plt.imshow(overlay); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(rgb);     plt.axis('off')
            plt.savefig(out); plt.close(fig)
        else:
            shutil.copy(path, os.path.join(out_dir, os.path.basename(path)))

    return cm, threshold, metrics

# Evaluation with segmentation (GPU-accelerated with full metrics)
def evaluation(encoder, bn, decoder, res, dataloader, device, img_path, eval_cfg=None):
    decoder.eval()
    bn.eval()
    eval_cfg = _resolve_eval_cfg(eval_cfg)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for img, gt, label, _, _ in dataloader:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # Optional dynamic crop (EvalConfig.dynamic_crop; the paper does not crop).
            # The crop's 0.94 background threshold is defined on [0,1] images, so
            # denormalize -> crop -> renormalize, only when requested.
            if eval_cfg.dynamic_crop:
                img = (img * std_t + mean_t).clamp(0, 1)
                img, gt = apply_dynamic_crop_gpu(img, masks=gt)
                img = (img - mean_t) / std_t

            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)

            anomaly_map, _ = compute_anomaly_map_torch(inputs[0:3], outputs, img.shape[-1], eval_cfg)
            gt = (gt > 0.5).float()
            
            # AUPRO Calculation (Requires CPU execution for regionprops)
            if label.item() != 0:
                gt_cpu = gt.squeeze().cpu().numpy().astype(int)
                amap_cpu = anomaly_map.squeeze().cpu().numpy()
                # Add np.newaxis to resolve the 3D shape assertion error (1, H, W)
                aupro_list.append(compute_pro(gt_cpu[np.newaxis, :, :], amap_cpu[np.newaxis, :, :]))
            
            # Tensors for Pixel-level metrics (Segmentation)
            gt_list_px.append(gt.view(-1))
            pr_list_px.append(anomaly_map.view(-1))
            
            # Tensors for Sample-level metrics (Image classification)
            gt_list_sp.append(label.item())
            pr_list_sp.append(anomaly_map.max().item())

    # Final synchronization with CPU only at the end of the epoch
    gt_px = torch.cat(gt_list_px).cpu().numpy().astype(int)
    pr_px = torch.cat(pr_list_px).cpu().numpy()
    
    gt_sp = np.array(gt_list_sp)
    pr_sp = np.array(pr_list_sp)
    
    # ---------------------------------------------------------
    # PIXEL-LEVEL METRICS (Defect Localization)
    # ---------------------------------------------------------
    auroc_px = round(roc_auc_score(gt_px, pr_px), 3)
    ap_loc = round(average_precision_score(gt_px, pr_px), 3)
    aupro = round(np.mean(aupro_list), 3) if len(aupro_list) > 0 else 0.0

    precisions_px, recalls_px, thresholds_px = precision_recall_curve(gt_px, pr_px)
    f1_scores_px = (2 * precisions_px * recalls_px) / (precisions_px + recalls_px + 1e-10)
    best_idx_px = min(np.argmax(f1_scores_px), len(thresholds_px) - 1)
    best_threshold_px = thresholds_px[best_idx_px]
    
    pr_px_binary = (pr_px >= best_threshold_px).astype(int)
    optimal_f1_px = round(f1_score(gt_px, pr_px_binary), 3)
    
    # ---------------------------------------------------------
    # SAMPLE-LEVEL METRICS (Image Classification)
    # ---------------------------------------------------------
    auroc_sp = round(roc_auc_score(gt_sp, pr_sp), 3)
    
    # Dynamically find the optimal threshold at the SAMPLE level
    precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    # Prevent index out of bounds if best_idx is the very last element
    best_idx = min(best_idx, len(thresholds) - 1) 
    best_threshold = thresholds[best_idx]
    
    pr_sp_binary = (pr_sp >= best_threshold).astype(int)
    
    optimal_f1_sp = round(f1_score(gt_sp, pr_sp_binary), 3)
    optimal_prec_sp = round(precision_score(gt_sp, pr_sp_binary), 3)
    optimal_rec_sp = round(recall_score(gt_sp, pr_sp_binary), 3)
    
    return auroc_px, auroc_sp, aupro, ap_loc, optimal_f1_sp, optimal_prec_sp, optimal_rec_sp, optimal_f1_px


# Evaluation with segmentation, very time-consuming
def evaluation_visA(encoder, bn, decoder, res, dataloader, device, img_path, eval_cfg=None):
    # VisA path: preprocessing untouched (no crop); only the score definition is configurable.
    decoder.eval()
    bn.eval()
    eval_cfg = _resolve_eval_cfg(eval_cfg)
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    # aupro_list = []
    with torch.no_grad():
        for img, gt, label, _, _ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res) 
            # Compute anomaly maps using encoder's first three outputs and decoder's outputs
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a', eval_cfg=eval_cfg)


            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.int()

            #unique_values = torch.unique(gt)
            #print("Unique values in gt:", unique_values)

            # if label.item() != 0:
            #     # print(gt.squeeze(0).cpu().numpy().astype(int))
            #     # print(set(gt.flatten()))
            #     aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
            #                                   anomaly_map[np.newaxis, :, :]))

            # Convert multi-dimensional arrays to one-dimensional arrays
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp#, round(np.mean(aupro_list), 3)