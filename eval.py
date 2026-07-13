import torch
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, accuracy_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score)
from test import compute_pro, apply_dynamic_crop_gpu, compute_anomaly_map_torch, image_score_from_map, image_score_corner_masked
from torch.utils.data import DataLoader
import json

from dataset.dataset import get_data_transforms, MVTecDataset, MVTecDataset_no_seg
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50

def compute_image_anomaly_score_and_map(inputs, outputs, image_size, border_margin=0, corner_size=0):
    """
    Computes the image-level anomaly score and spatial map using the CANONICAL
    definition shared with test.py (and baked into the ONNX export): sum of
    per-layer (1 - cosine similarity) maps upsampled with align_corners=False,
    Gaussian blur k=15 sigma=4 with zero padding — the exact kernel used for
    model selection during training. Score = max of the BLURRED map.

    border_margin: pixel esclusi dall'INTERA fascia di bordo (uniforme sui 4 lati).
    corner_size: pixel esclusi SOLO nei 4 quadrati d'angolo (piu' chirurgico).
    Usa uno dei due, non entrambi insieme. Default 0/0 = comportamento invariato.
    """
    anomaly_map, _ = compute_anomaly_map_torch(inputs[0:3], outputs, image_size)
    if corner_size > 0:
        score = image_score_corner_masked(anomaly_map, corner_size=corner_size).item()
    else:
        score = image_score_from_map(anomaly_map, border_margin=border_margin).item()
    return score, anomaly_map

def save_confusion_map(img_tensor, mask_tensor, anomaly_map_tensor, save_path, global_min=0.0, global_max=1.0, threshold=0.5):
    """
    Generates a figure containing Original Image, GT Mask, and Anomaly Overlay.
    Uses Image-Relative Threshold-Centric Normalization:
    - Values below the decision threshold map to [0.0, 0.5] (Strictly Cold colors: Blue/Cyan)
    - Values above the decision threshold map to [0.5, 1.0] (Warm colors).
    - The maximum anomaly value IN THIS SPECIFIC IMAGE is forced to 1.0 (Deep Red).
    """
    # Denormalize image assuming [0, 1] range from transforms.ToTensor()
    img = img_tensor.squeeze().permute(1, 2, 0).numpy()
    if img.min() < 0:
        # Revert standard ImageNet normalization if applied
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Process anomaly map (RAW values)
    amap = anomaly_map_tensor.squeeze().numpy()
    
    # Image-specific maximum
    image_max = amap.max()
    
    # Prepare the colored map
    amap_colored = np.zeros_like(amap)
    
    # Normal pixels (Below Threshold) -> Scale relative to the global dataset baseline
    # This ensures that a normal image looks completely blue, exactly like the background of a defective image
    mask_normal = amap < threshold
    if (threshold - global_min) > 0:
        amap_colored[mask_normal] = 0.5 * ((amap[mask_normal] - global_min) / (threshold - global_min))
        
    # Anomalous pixels (Above Threshold) -> Scale relative to THIS IMAGE's peak
    # This guarantees that the worst defect in this specific image hits 1.0 (Deep Red)
    mask_anomalous = amap >= threshold
    if image_max > threshold:
        # Stretch values between threshold (maps to 0.5) and image_max (maps to 1.0)
        amap_colored[mask_anomalous] = 0.5 + 0.5 * ((amap[mask_anomalous] - threshold) / (image_max - threshold))
    elif np.any(mask_anomalous):
         # Edge case: pixel is exactly at threshold and is the max value
         amap_colored[mask_anomalous] = 0.5
         
    # Strict clipping to avoid colormap artifacts
    amap_colored = np.clip(amap_colored, 0, 1)
    
    # Create colormap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * amap_colored), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Adjust overlay weights to make the heatmap punchier
    overlay = 0.4 * img + 0.6 * heatmap
    
    # Setup plotting
    has_mask = mask_tensor is not None
    fig, axes = plt.subplots(1, 3 if has_mask else 2, figsize=(12 if has_mask else 8, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if has_mask:
        gt = mask_tensor.squeeze().numpy()
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Anomaly Heatmap')
        axes[2].axis('off')
    else:
        axes[1].imshow(overlay)
        axes[1].set_title('Anomaly Heatmap')
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def evaluate_and_save_maps(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    image_size = 256
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = os.path.join(args.data_path, args.class_)
    
    if args.seg == 0:
        test_data = MVTecDataset_no_seg(root=test_path, transform=data_transform, phase="test")
    else:
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    # Initialize networks
    if args.net == 'wide_res50':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif args.net == 'res18':
        encoder, bn = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False)
    elif args.net == 'res34':
        encoder, bn = resnet34(pretrained=True)
        decoder = de_resnet34(pretrained=False)
    elif args.net == 'res50':
        encoder, bn = resnet50(pretrained=True)
        decoder = de_resnet50(pretrained=False)
    
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()
    bn.eval()
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        if 'bn' in checkpoint:
            bn.load_state_dict(checkpoint['bn'])
    else:
        decoder.load_state_dict(checkpoint)

    # Prepare directories for Confusion Matrix categories
    categories = ['tp', 'tn', 'fp', 'fn']
    for cat in categories:
        os.makedirs(os.path.join(args.img_path, cat), exist_ok=True)

    results_memory = []

    # Tensors for the canonical denormalize -> crop -> renormalize step
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print("Phase 1/2: Extracting anomaly scores and spatial maps...")
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            img = batch[0].to(device)
            
            label = None
            mask = None
            
            # Dynamically inspect all items returned by the dataloader to find mask and label
            for item in batch[1:]:
                if isinstance(item, torch.Tensor):
                    if item.numel() == 1:
                        # It's a scalar tensor -> label
                        label = item.item()
                    elif item.numel() > 1:
                        # It's a 2D+ tensor -> mask
                        mask = item.to(device)
                elif isinstance(item, (int, float)):
                    # It's a standard number -> label
                    label = int(item)
            
            # Fallback: if the dataset only returned (image, mask) without an explicit label,
            # we infer the label directly from the mask (all-zero mask = 0, otherwise = 1)
            if label is None:
                if mask is not None:
                    label = 1 if mask.max().item() > 0 else 0
                else:
                    raise RuntimeError("Dataset returned neither a label nor a mask.")
            
            # Canonical preprocessing, identical to the training loop (main.py)
            # and to what the inference runtime does: the dynamic crop operates
            # on the DENORMALIZED [0,1] image (its 0.94 background threshold is
            # defined there), then the crop is renormalized for the encoder.
            # Without this crop the threshold computed below would live on a
            # different score distribution than production inference.
            img = (img * std_t + mean_t).clamp(0, 1)
            if mask is not None:
                img, mask = apply_dynamic_crop_gpu(img, masks=mask)
            else:
                img = apply_dynamic_crop_gpu(img)
            img = (img - mean_t) / std_t

            # Forward pass
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], args.res)
            
            score, anomaly_map = compute_image_anomaly_score_and_map(inputs, outputs, image_size, border_margin=args.border_margin, corner_size=args.corner_size)
            
            # Store everything on CPU to avoid GPU OOM on large datasets
            results_memory.append({
                'idx': idx,
                'img': img.cpu(),
                'mask': mask.cpu() if mask is not None else None,
                'anomaly_map': anomaly_map.cpu(),
                'label': label,
                'score': score
            })

    # Extract raw arrays for image-level evaluation
    y_true = np.array([r['label'] for r in results_memory])
    raw_scores = np.array([r['score'] for r in results_memory])

    # Calculate global spatial map limits for visualization normalization
    global_max = raw_scores.max()
    global_min = min([r['anomaly_map'].min().item() for r in results_memory])

    # ---------------------------------------------------------
    # SAMPLE-LEVEL METRICS (Image Classification)
    # ---------------------------------------------------------
    # Dynamically find the optimal threshold using F1-Score maximization
    precisions, recalls, thresholds = precision_recall_curve(y_true, raw_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold_raw = thresholds[best_idx]
    
    # Generate binary predictions based on the optimal threshold
    y_pred = (raw_scores >= best_threshold_raw).astype(int)
    
    # Calculate sample-level classification metrics
    auroc_sp = roc_auc_score(y_true, raw_scores)
    cm = confusion_matrix(y_true, y_pred)
    f1_sp = f1_score(y_true, y_pred)
    prec_sp = precision_score(y_true, y_pred)
    rec_sp = recall_score(y_true, y_pred)
    acc_sp = accuracy_score(y_true, y_pred)

    # ---------------------------------------------------------
    # PIXEL-LEVEL METRICS (Defect Localization)
    # ---------------------------------------------------------
    if args.seg == 1:
        print("Calculating pixel-level metrics (AUPRO, AP-loc, AUROC). This may take a moment...")
        gt_list_px = []
        pr_list_px = []
        aupro_list = []
        
        for res in results_memory:
            if res['mask'] is not None:
                # Extract and binarize the ground truth mask
                mask_np = res['mask'].squeeze().numpy()
                amap_np = res['anomaly_map'].squeeze().numpy()
                mask_binary = (mask_np > 0.5).astype(int)
                
                # Flatten the 2D arrays to 1D for sklearn compatibility
                gt_list_px.extend(mask_binary.ravel())
                pr_list_px.extend(amap_np.ravel())
                
                # AUPRO is evaluated only on images containing actual anomalies
                if res['label'] == 1:
                    pro_score = compute_pro(mask_binary[np.newaxis, :, :], amap_np[np.newaxis, :, :])
                    aupro_list.append(pro_score)
                    
        # Convert lists to NumPy arrays
        gt_px = np.array(gt_list_px)
        pr_px = np.array(pr_list_px)
        
        # Calculate AUC and Average Precision metrics at pixel level
        auroc_px = roc_auc_score(gt_px, pr_px)
        ap_loc = average_precision_score(gt_px, pr_px)
        aupro = np.mean(aupro_list) if len(aupro_list) > 0 else 0.0

    # ---------------------------------------------------------
    # PRINT EVALUATION REPORT
    # ---------------------------------------------------------
    print("=" * 50)
    print(" EVALUATION METRICS REPORT ")
    print("=" * 50)
    print(f"border_margin={args.border_margin}  corner_size={args.corner_size}  "
          f"({'comportamento invariato' if args.border_margin == 0 and args.corner_size == 0 else 'ATTIVO'})")
    print("-" * 50)

    print("--- SAMPLE LEVEL (Image Classification) ---")
    print(f"Optimal Threshold: {best_threshold_raw:.4f}")
    print(f"AUROC:             {auroc_sp:.4f}")
    print(f"Accuracy:          {acc_sp:.4f}")
    print(f"F1-Score:          {f1_sp:.4f}")
    print(f"Precision:         {prec_sp:.4f}")
    print(f"Recall:            {rec_sp:.4f}")
    print("Confusion Matrix (TN, FP | FN, TP):")
    print(cm)
    
    if args.seg == 1:
        print("\n--- PIXEL LEVEL (Defect Localization) ---")
        print(f"AUPRO:             {aupro:.4f}")
        print(f"AP-loc:            {ap_loc:.4f}")
        print(f"Pixel AUROC:       {auroc_px:.4f}")
    print("=" * 50)

    # Persist the calibration so it can travel with the model and be
    # cross-checked against calibrate_threshold.py (the ONNX-pipeline
    # calibration): with the canonical pipeline the two must agree up to
    # floating-point drift.
    calibration = {
        "pipeline": "dynamic_crop -> sum(1-cos, align_corners=False) -> gauss_k15_sigma4_zeropad -> max",
        "source": "eval.py (PyTorch reference)",
        "checkpoint": os.path.basename(args.checkpoint_path),
        "class": args.class_,
        "threshold": float(best_threshold_raw),
        "global_min": float(global_min),
        "global_max": float(global_max),
        "auroc_sp": float(auroc_sp),
        "f1_sp": float(f1_sp),
    }
    calib_path = os.path.join(args.img_path, "calibration_pytorch.json")
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Calibration saved to {calib_path}")

    print("\nPhase 2/2: Saving visualizations categorized by Confusion Matrix...")
    for res in results_memory:
        # Use raw score and raw threshold for categorization
        pred = 1 if res['score'] >= best_threshold_raw else 0
        actual = res['label']
        
        if actual == 1 and pred == 1:
            category = 'tp'
        elif actual == 0 and pred == 0:
            category = 'tn'
        elif actual == 0 and pred == 1:
            category = 'fp'
        else:
            category = 'fn'
            
        save_path = os.path.join(args.img_path, category, f"sample_{res['idx']:04d}_score_{res['score']:.4f}.png")
        
        # Pass RAW values to the visualization function
        # The function will internally handle scaling the background to blue 
        # and forcing the image's specific defect to deep red.
        save_confusion_map(res['img'], res['mask'], res['anomaly_map'], save_path, global_min, global_max, best_threshold_raw)
        
    print(f"Done! Evaluated images sorted in {args.img_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and categorize Anomaly Maps")
    parser.add_argument('--class_', default='bottle', type=str, help='Dataset class to evaluate')
    parser.add_argument('--data_path', default='./mvtec/', type=str, help='Path to dataset root')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the .pth model file')
    parser.add_argument('--img_path', default='./results_maps/', type=str, help='Output path for categorized heatmaps')
    parser.add_argument('--seg', default=1, type=int, help='0 for no segmentation, 1 with segmentation masks')
    parser.add_argument('--res', default=3, type=int, help='Skip connection parameter used during training')
    parser.add_argument('--net', default='wide_res50', type=str, help='Network architecture')
    parser.add_argument('--border_margin', default=0, type=int,
                         help='Pixel esclusi dall\'intera fascia di bordo nel calcolo dello score. '
                              '0 = comportamento invariato. DA VALIDARE.')
    parser.add_argument('--corner_size', default=0, type=int,
                         help='Pixel esclusi SOLO nei 4 angoli (piu\' chirurgico di --border_margin). '
                              '0 = comportamento invariato. DA VALIDARE.')
    
    args = parser.parse_args()
    evaluate_and_save_maps(args)