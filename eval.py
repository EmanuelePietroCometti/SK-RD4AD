import torch
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, accuracy_score, confusion_matrix, 
                             precision_recall_curve)
from torch.utils.data import DataLoader

from dataset.dataset import get_data_transforms, MVTecDataset, MVTecDataset_no_seg
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50

def compute_image_anomaly_score_and_map(inputs, outputs, image_size):
    """
    Computes the image-level anomaly score and spatial map using Cosine Similarity.
    Applies Gaussian smoothing BEFORE extracting the max score to ensure stability.
    """
    anomaly_map = torch.zeros(1, 1, image_size, image_size).to(inputs[0].device)
    
    for i in range(len(outputs)):
        a = inputs[i]
        b = outputs[i]
        
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
        
        # Compute Cosine Similarity
        distance = 1 - torch.sum(a_norm * b_norm, dim=1, keepdim=True)
        
        # Interpolate to original image size
        distance = F.interpolate(distance, size=(image_size, image_size), 
                                 mode='bilinear', align_corners=False)
        anomaly_map += distance
        
    # Move to CPU and convert to Numpy for OpenCV processing
    amap_np = anomaly_map.squeeze().cpu().numpy()
    
    # Apply Gaussian Blur using a 15x15 kernel (standard for 256x256 resolution)
    amap_np = cv2.GaussianBlur(amap_np, (15, 15), 0)
    
    # Convert back to tensor to maintain compatibility with the pipeline
    anomaly_map = torch.from_numpy(amap_np).unsqueeze(0).unsqueeze(0).to(inputs[0].device)
    
    # The extracted max() is now consistent with the saved anomaly map
    return anomaly_map.max().item(), anomaly_map

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
    
    # 1. Normal pixels (Below Threshold) -> Scale relative to the global dataset baseline
    # This ensures that a normal image looks completely blue, exactly like the background of a defective image
    mask_normal = amap < threshold
    if (threshold - global_min) > 0:
        amap_colored[mask_normal] = 0.5 * ((amap[mask_normal] - global_min) / (threshold - global_min))
        
    # 2. Anomalous pixels (Above Threshold) -> Scale relative to THIS IMAGE's peak
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
        encoder = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False)
    elif args.net == 'res34':
        encoder = resnet34(pretrained=True)
        decoder = de_resnet34(pretrained=False)
    elif args.net == 'res50':
        encoder = resnet50(pretrained=True)
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
            
            # Forward pass
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], args.res)
            
            score, anomaly_map = compute_image_anomaly_score_and_map(inputs, outputs, image_size)
            
            # Store everything on CPU to avoid GPU OOM on large datasets
            results_memory.append({
                'idx': idx,
                'img': img.cpu(),
                'mask': mask.cpu() if mask is not None else None,
                'anomaly_map': anomaly_map.cpu(),
                'label': label,
                'score': score
            })

    # Extract raw arrays
    y_true = np.array([r['label'] for r in results_memory])
    raw_scores = np.array([r['score'] for r in results_memory])

    # Calculate GLOBAL spatial map min and max FROM RAW DATA
    global_max = raw_scores.max()
    global_min = min([r['anomaly_map'].min().item() for r in results_memory])

    # Find optimal threshold using F1-Score maximization on RAW scores
    precisions, recalls, thresholds = precision_recall_curve(y_true, raw_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold_raw = thresholds[best_idx]
    
    # Generate predictions based on the raw threshold
    y_pred = (raw_scores >= best_threshold_raw).astype(int)
    
    # Calculate metrics
    auroc = roc_auc_score(y_true, raw_scores)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 40)
    print(" EVALUATION METRICS REPORT ")
    print("=" * 40)
    print(f"Optimal Threshold (RAW): {best_threshold_raw:.4f}")
    print(f"AUROC:             {auroc:.4f}")
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score:          {f1_score(y_true, y_pred):.4f}")
    print(f"Precision:         {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:            {recall_score(y_true, y_pred):.4f}")
    print("-" * 40)
    print("Confusion Matrix (TN, FP | FN, TP):")
    print(cm)
    print("=" * 40)

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
    
    args = parser.parse_args()
    evaluate_and_save_maps(args)