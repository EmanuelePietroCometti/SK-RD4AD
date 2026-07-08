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
                             precision_score, recall_score)
import matplotlib
import pickle
import os
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import normalize
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F_v2

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

# Calculate anomaly score map (Optimized for GPU)
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    # Get the device and batch size dynamically from the input tensors
    device = fs_list[0].device
    batch = fs_list[0].shape[0]

    # Initialize the base anomaly map directly on the GPU
    if amap_mode == 'mul':
        anomaly_map = torch.ones((batch, 1, out_size, out_size), device=device)
    else:
        anomaly_map = torch.zeros((batch, 1, out_size, out_size), device=device)
        
    a_map_list = []
    
    for i in range(len(ft_list)):  # Iterate over anomaly maps
        fs = fs_list[i]
        ft = ft_list[i]
        
        # Compute cosine similarity on GPU
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1) 
        a_map = torch.unsqueeze(a_map, dim=1)  # Shape: (B, 1, H, W)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) 
        
        # Perform accumulation (multiply or add) directly on GPU
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
            
        # Store individual maps on CPU only if needed for return list
        a_map_list.append(a_map.squeeze().cpu().detach().numpy())
        
    kernel_size = 15
    sigma = 4
    x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    kernel = (gauss[:, None] * gauss[None, :])
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    
    # Pad the tensor to maintain the original spatial dimensions
    anomaly_map = F.conv2d(anomaly_map, kernel, padding=kernel_size//2)

    # Move the final combined map to CPU ONCE at the very end
    final_anomaly_map = anomaly_map.squeeze().cpu().detach().numpy()
    
    return final_anomaly_map, a_map_list

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
def evaluation_me(encoder, bn, decoder, res, dataloader, device, print_canshu, score_num):
    decoder.eval() 
    bn.eval()
    encoder.eval()
    
    # Lists to store sample-level labels and predictions
    gt_list_sp = [] 
    pr_list_sp = [] 


    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for (img, label, _) in dataloader:
            img = img.to(device)

            # Denormalize, Crop, Renormalize
            img = (img * std_t + mean_t).clamp(0, 1)
            img = apply_dynamic_crop_gpu(img)
            img = (img - mean_t) / std_t

            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)

            # Calculate final anomaly map (supports any batch size)
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
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

# Generate heatmaps for evaluation visualization
def evaluation_visualization(encoder, bn, decoder, res, dataloader, device, print_canshu, score_num, img_path):
    count = 0
    decoder.eval()
    bn.eval()
    with torch.no_grad():
        for img, gt, label, _, ip in dataloader:
            print(ip[0][-20:-4])
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)  

            anomaly_map, amap_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')  # Generate anomaly map
            ano_map = min_max_norm(anomaly_map)  # Normalize data

            ano_map = cvt2heatmap(255-ano_map*255)  # Convert to heatmap
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)

            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)  # Overlay heatmap on original image

            # Plot heatmap
            plt.subplot(1,3,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # Plot ground truth
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            plt.subplot(1,3,2)
            plt.imshow(gt, cmap='gray')
            plt.axis('off')

            # Plot original image
            plt.subplot(1,3,3)
            plt.imshow(img)
            plt.axis('off')

            if (os.path.exists(img_path) == 0):
                os.mkdir(img_path)

            # Save image
            original_name = os.path.basename(ip[0])
            
            # Split the name from its extension and append '.png'
            name_without_ext = os.path.splitext(original_name)[0]
            file_name_png = name_without_ext + '.png'
            
            # Save safely as a PNG file
            plt.savefig(os.path.join(img_path, file_name_png))

            count += 1

# Generate heatmaps for evaluation visualization without segmentation
def evaluation_visualization_no_seg(encoder, bn, decoder, res, dataloader, device, print_canshu, score_num, img_path):
    count = 0
    decoder.eval()
    bn.eval()
    with torch.no_grad():
        for img, label, _  in dataloader:
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)  

            anomaly_map, amap_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')  # Generate anomaly map
            
            ano_map = min_max_norm(anomaly_map)  # Normalize data

            ano_map = cvt2heatmap(255-ano_map*255)  # Convert to heatmap
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)

            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)  # Overlay heatmap on original image

            # Plot heatmap
            plt.subplot(1,2,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # Plot original image
            plt.subplot(1,2,2)
            plt.imshow(img)
            plt.axis('off')

            if (os.path.exists(img_path) == 0):
                os.mkdir(img_path)

            # Save image
            plt.savefig(img_path + str(count).replace('/', '_') + '.png')

            count += 1

# Evaluation with segmentation (GPU-accelerated with full metrics)
def evaluation(encoder, bn, decoder, res, dataloader, device, img_path):
    decoder.eval()
    bn.eval()
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    
    # Pre-define Gaussian kernel on GPU for speed optimization
    kernel_size = 15
    sigma = 4
    x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    kernel = (gauss[:, None] * gauss[None, :])
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)

    with torch.no_grad():
        for img, gt, label, _, _ in dataloader:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            # Dynamic crop executed on GPU
            img, gt = apply_dynamic_crop_gpu(img, masks=gt)
            
            inputs = encoder(img)
            outputs = decoder(bn(inputs), inputs[0:3], res)
            
            # Anomaly Map Calculation on GPU
            anomaly_map = torch.zeros((1, 1, 256, 256), device=device)
            for i in range(len(outputs)):
                dist = 1 - F.cosine_similarity(inputs[i], outputs[i], dim=1).unsqueeze(1)
                anomaly_map += F.interpolate(dist, size=256, mode='bilinear', align_corners=False)
            
            # Apply Gaussian blur on GPU
            anomaly_map = F.conv2d(anomaly_map, kernel, padding=kernel_size//2)
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
def evaluation_visA(encoder, bn, decoder, res, dataloader, device, img_path):
    decoder.eval()
    bn.eval()
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
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
            

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