import torch
from dataset.dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import random
import os
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset.dataset import MVTecDataset, MVTecDataset_no_seg
import torch.backends.cudnn as cudnn
import argparse
import sys
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F_v2
import torch.nn.functional as F
import cv2

from test import evaluation_me, evaluation_visualization, evaluation, evaluation_visualization_no_seg, apply_dynamic_crop_gpu
from dust_contrastive import DustBank, ProjectionHead, dust_contrastive_loss

def raw_tensor_loader(path):
    # Read the image using OpenCV (loads in BGR format by default)
    # This operation is highly optimized and releases the Python GIL
    img_bgr = cv2.imread(path)
    
    if img_bgr is None:
        print(f"Warning: OpenCV failed to load {path}. Returning empty tensor.")
        # Fallback tensor to prevent training crashes on a corrupted image
        return torch.zeros((3, 256, 256), dtype=torch.uint8)
        
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array (H, W, C) to PyTorch tensor (C, H, W)
    # contiguous() ensures memory alignment for optimal GPU transfer
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    
    return tensor

# Set random seed
def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loss function, can be used for ablation studies.
# Returns only the main distillation loss; the inter-group consistency loss
# (loss_function_2) is computed separately, only when layerloss == 1.
def loss_function(a, b, L2):  # Input two tensor arrays
    cos_loss = torch.nn.CosineSimilarity()
    #print(a[0].size())  # a[0] = [16,256,64,64]
    #print(a[1].size())  # a[1] = [16,512,32,32]
    #print(a[2].size())  # a[2] = [16,1024,16,16]
    loss = 0

    # Use cosine loss
    if L2 == 0:
        for item in range(len(a)):  # For each tensor in a
            loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))

    # Use L2 loss + cosine loss
    if L2 == 2:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += 0.5 * torch.mean(l2_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
             loss += 0.5 * torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))

    # Use L2 loss
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
    return loss

# Try to calculate inter-group consistency loss
def loss_function_2(a, b):  # Input two tensor arrays
    mse_loss = torch.nn.MSELoss()
    # Compare results obtained by upsampling a2 and b2 with results obtained without upsampling a1 and b1
    a2 = F.interpolate(a[2], size=32, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=32, mode='bilinear', align_corners=True)
    l2 = torch.mean(mse_loss(a2.view(a2.shape[0],-1), b2.view(b2.shape[0],-1)))
    l1 = torch.mean(mse_loss(a[1].view(a[1].shape[0],-1), b[1].view(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l2-l1)

    # Compare results obtained by upsampling a1 and b1 with results obtained without upsampling a0 and b0
    l0 = torch.mean(mse_loss(a[0].view(a[0].shape[0],-1), b[0].view(b[0].shape[0],-1)))

    a1 = F.interpolate(a[1], size=64, mode='bilinear', align_corners=True)
    b1 = F.interpolate(b[1], size=64, mode='bilinear', align_corners=True)
    l1 = torch.mean(mse_loss(a1.view(a1.shape[0],-1), b1.view(b1.shape[0],-1)))
    loss2_2 = torch.abs(l1-l0)

    #print(loss2_1,loss2_2)
    #sys.exit()
    loss2 = loss2_1 + loss2_2
    return loss2

def train(class_, epochs, learning_rate, res, batch_size, print_epoch, seg, data_path, save_path, print_canshu, score_num, print_loss, img_path, vis, cut, layerloss, rate, print_max, net, L2, seed,
          contrastive=0, contrastive_rate=0.1, contrastive_temp=0.1, dust_bank_path=None):
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(class_)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data_transform, gt_transform = get_data_transforms(image_size, image_size) 
    data_transform = data_transform.to(device)
    gt_transform = gt_transform.to(device)

    train_path = data_path + class_ + '/train'
    test_path = data_path + class_ 
    ckp_path = save_path + net + class_ 

    train_data = ImageFolder(root=train_path, loader=raw_tensor_loader)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    data_transforms, _ = get_data_transforms(size=image_size, isize=image_size)

    # Move transforms to GPU (useful for certain v2 operations)
    data_transforms = data_transforms.to(device)

    # Whether to use segmentation
    if seg == 0:  
        test_data = MVTecDataset_no_seg(root=test_path, transform=data_transforms, phase="test")
        # evaluation_me is vectorized: batching the test set speeds up evaluation
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if seg == 1:
        test_data = MVTecDataset(root=test_path, transform=data_transforms, gt_transform=gt_transform, phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Choose which network to use
    if net == 'wide_res50':   
        encoder, bn = wide_resnet50_2(pretrained=True)  # Encoder and bottleneck
        encoder = encoder.to(device)
        bn = bn.to(device)
        encoder.eval()  # Freeze encoder model parameters    
        decoder = de_wide_resnet50_2(pretrained=False)  # Decoder initialization
        decoder = decoder.to(device)
        
    if net == 'res18':
        encoder, bn = resnet18(pretrained=True)  # Unpack encoder and bottleneck
        encoder = encoder.to(device)
        bn = bn.to(device)  # Move bottleneck to device
        encoder.eval()  # Freeze encoder model parameters    
        decoder = de_resnet18(pretrained=False)  # Decoder initialization
        decoder = decoder.to(device)
        
    if net == 'res34':
        encoder, bn = resnet34(pretrained=True)  # Unpack encoder and bottleneck
        encoder = encoder.to(device)
        bn = bn.to(device)  # Move bottleneck to device
        encoder.eval()  # Freeze encoder model parameters    
        decoder = de_resnet34(pretrained=False)  # Decoder initialization
        decoder = decoder.to(device)
        
    if net == 'res50':
        encoder, bn = resnet50(pretrained=True)  # Unpack encoder and bottleneck
        encoder = encoder.to(device)
        bn = bn.to(device)  # Move bottleneck to device
        encoder.eval()  # Freeze encoder model parameters    
        decoder = de_resnet50(pretrained=False)  # Decoder initialization
        decoder = decoder.to(device)

    # ==========================================
    # DUST CONTRASTIVE LOSS: setup (branch contrastive_loss)
    # ==========================================
    dust_bank = None
    proj_head = None
    if contrastive == 1:
        if not dust_bank_path:
            raise ValueError("--contrastive 1 richiede --dust_bank_path (esegui prima verify_dust_pipeline.py).")
        if batch_size < 2:
            raise ValueError("--contrastive 1 richiede batch_size >= 2 (serve un'immagine 'donor' diversa "
                              "da se stessa per generare i pseudo-difetti CutPaste/Scar).")
        dust_bank = DustBank(dust_bank_path, device)

        # Rileva a runtime il numero di canali dell'embedding OCBE (bn(inputs)): dipende da --net
        # (2048 per wide_res50/res50 con Bottleneck, 512 per res18/res34 con BasicBlock). Evitiamo
        # di hardcodarlo per non rompere silenziosamente la ProjectionHead cambiando --net.
        with torch.no_grad():
            dummy = torch.zeros(2, 3, image_size, image_size, device=device)
            dummy_inputs = encoder(dummy)
            dummy_embed = bn(dummy_inputs)
            ocbe_channels = dummy_embed.shape[1]
        proj_head = ProjectionHead(in_channels=ocbe_channels).to(device)
        print(f"[contrastive] OCBE embedding channels = {ocbe_channels}, "
              f"contrastive_rate = {contrastive_rate}, contrastive_temp = {contrastive_temp}")

    optimizer_params = list(decoder.parameters()) + list(bn.parameters())
    if proj_head is not None:
        optimizer_params += list(proj_head.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate, betas=(0.5,0.999))  # Pass a list of parameters to be optimized

    max_auc = []
    max_auc_epoch = []
    max_pr = []
    max_pr_epoch = []
    best_avg_score = 0
    best_metrics = None  # stays None if no evaluation ever improves the score

    # Define v2 pipeline (executed directly on GPU tensors)
    gpu_transforms = v2.Compose([
        v2.RandomAffine(degrees=[-10.0, 10.0], translate=[0.02, 0.02], scale=[0.98, 1.02], fill=1.0, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.4)
    ])

    # Tensors for denormalization/normalization on device
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Start training
    for epoch in range(epochs):
        decoder.train()
        bn.train()
        if proj_head is not None:
            proj_head.train()
        loss_list = []
        loss_recon_list, loss_dust_recon_list, loss_contrastive_list = [], [], []
        sim_pos_list, sim_neg_list = [], []
        for img, label in train_dataloader:
            img = img.to(device, non_blocking=True) 
            img = data_transform(img)

            # ==========================================
            # INLINE GPU AUGMENTATION BLOCK
            # ==========================================
            # Denormalize to [0, 1] range for color operations
            img = img * std_t + mean_t
            img = img.clamp(0, 1)
            
            # Apply Dynamic Crop (Always active to normalize object scale)
            img = apply_dynamic_crop_gpu(img)

            # Apply Equalization (50% probability) — CPU rng, avoids a GPU sync per batch
            if random.random() < 0.5:
                img_uint8 = (img * 255.0).to(torch.uint8)
                img = F_v2.equalize(img_uint8).to(torch.float32) / 255.0

            # Standard spatial and color transformations
            img = gpu_transforms(img)

            # ---- DUST CONTRASTIVE LOSS: genera le view a partire dall'immagine pulita ----
            # Va fatto QUI: img è ancora in [0,1] e ha già subito crop/augmentation geometriche,
            # cosi' la polvere/il pseudo-difetto vengono incollati in coordinate coerenti con
            # l'oggetto (non rischiano di finire fuori dal crop dinamico).
            if contrastive == 1:
                img_p, img_n = dust_bank.build_views(img)
                img_p = (img_p - mean_t) / std_t
                img_n = (img_n - mean_t) / std_t

            # Renormalize back to ImageNet standard for ResNet encoder
            img = (img - mean_t) / std_t
            # ==========================================

            # The encoder is frozen: skip autograd graph construction for it
            # (saves memory and compute; gradients only flow through bn/decoder)
            if contrastive == 1:
                combined = torch.cat([img, img_p, img_n], dim=0)
            else:
                combined = img

            with torch.no_grad():
                inputs_all = encoder(combined)

            if contrastive == 1:
                Bsz = img.shape[0]
                inputs = [t[:Bsz] for t in inputs_all]
                inputs_p = [t[Bsz:2 * Bsz] for t in inputs_all]
                inputs_n = [t[2 * Bsz:] for t in inputs_all]
            else:
                inputs = inputs_all

            embed_a = bn(inputs)
            outputs = decoder(embed_a, inputs[0:3], res)

            # Choose loss function
            recon_loss = loss_function(inputs[0:3], outputs, L2)
            loss = recon_loss
            if layerloss == 1:
                loss = loss + rate * loss_function_2(inputs[0:3], outputs)

            dust_recon_loss = torch.tensor(0.0, device=device)
            c_loss = torch.tensor(0.0, device=device)
            if contrastive == 1:
                # Il decoder deve imparare a ricostruire correttamente anche le feature "sporche"
                # di polvere: senza questo termine, la mappa di anomalia pixel-level resterebbe
                # comunque alta sulla polvere in inferenza, indipendentemente da quanto sia pulito
                # lo spazio dell'embedding grazie alla contrastive loss (che agisce solo sul
                # bottleneck, non sulla ricostruzione pixel-level).
                embed_p = bn(inputs_p)
                outputs_p = decoder(embed_p, inputs_p[0:3], res)
                dust_recon_loss = loss_function(inputs_p[0:3], outputs_p, L2)
                loss = loss + dust_recon_loss

                # Non serve il decoder per il negativo: ci serve solo l'embedding OCBE
                embed_n = bn(inputs_n)

                z_a = proj_head(embed_a)
                z_p = proj_head(embed_p)
                z_n = proj_head(embed_n)
                c_loss = dust_contrastive_loss(z_a, z_p, z_n, temperature=contrastive_temp)
                loss = loss + contrastive_rate * c_loss

                # Diagnostica scale-free (NON dipende da contrastive_temp, a differenza di c_loss):
                # e' la similarita' coseno grezza, l'unica cosa che dice davvero "quanto sono
                # separati" anchor/polvere vs anchor/pseudo-difetto nello spazio dell'embedding.
                with torch.no_grad():
                    sim_pos_mean = (z_a * z_p).sum(dim=1).mean().item()
                    sim_neg_mean = (z_a * z_n).sum(dim=1).mean().item()

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_recon_list.append(recon_loss.item())
            if contrastive == 1:
                loss_dust_recon_list.append(dust_recon_loss.item())
                loss_contrastive_list.append(c_loss.item())
                sim_pos_list.append(sim_pos_mean)
                sim_neg_list.append(sim_neg_mean)

        if print_loss == 1:
            if contrastive == 1:
                print('epoch [{}/{}], loss_totale:{:.4f}  (recon_clean:{:.4f}  recon_dust:{:.4f}  contrastive:{:.4f})'.format(
                    epoch + 1, epochs, np.mean(loss_list), np.mean(loss_recon_list),
                    np.mean(loss_dust_recon_list), np.mean(loss_contrastive_list)))
                print('           cos_sim(anchor,polvere)={:.3f}  cos_sim(anchor,difetto)={:.3f}  gap={:.3f}'.format(
                    np.mean(sim_pos_list), np.mean(sim_neg_list), np.mean(sim_pos_list) - np.mean(sim_neg_list)))
            else:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % print_epoch == 0:
            # Test set without mask
            if seg == 0:
                auroc_sp= evaluation_me(encoder,bn, decoder, res, test_dataloader, device, print_canshu, score_num)
                print('epoch:', (epoch + 1))
                print('Sample Auroc{:.3f}'.format(auroc_sp))
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch + 1)
                if print_max == 1:
                    print('max_auc = ', max(max_auc))
                    print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('------------------')

                # Save model only if Sample AUROC is the maximum
                current_auroc_score = auroc_sp

                if current_auroc_score > best_avg_score:
                    print(f"New best model found at epoch {epoch+1} with Sample Auroc{auroc_sp:.3f}")
                    torch.save({'bn': bn.state_dict(),'decoder': decoder.state_dict()}, ckp_path + str(epoch+1) + str(seed) + 'sample_auc=' + str(auroc_sp) + '.pth')
                    best_avg_score = current_auroc_score
                    best_metrics = (auroc_sp,)
               
                if vis == 1:  # Visualization output when no mask
                    evaluation_visualization_no_seg(encoder, bn, decoder, res, test_dataloader, device, print_canshu, score_num, img_path)

            # Test set with mask and need localization
            if seg == 1:
                # Go through normal process
                # Plot
                if vis == 1:
                    evaluation_visualization(encoder, bn, decoder, res, test_dataloader, device, print_canshu, score_num, img_path)
                # This part calculates the basic results and saves the results of the current epoch.
                auroc_px, auroc_sp, aupro, ap_loc, f1, prec, rec, f1_px = evaluation(encoder, bn, decoder, res, test_dataloader, device, img_path)
                
                print(f'Pixel AUROC: {auroc_px:.3f}, Sample AUROC: {auroc_sp:.3f}, AUPRO: {aupro:.3f}')
                print(f'AP-loc: {ap_loc:.3f}, F1-Score: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1-px: {f1_px:.3f}')


                # Update AUROC and AUPRO lists
                max_auc.append(auroc_px)
                max_auc_epoch.append(epoch + 1)
                max_pr.append(aupro)
                max_pr_epoch.append(epoch + 1)


                # Print maximum AUROC and AUPRO, and the corresponding epoch
                print('max_auc = ', max(max_auc))
                print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('max_pr = ', max(max_pr))
                print('max_epoch = ', max_pr_epoch[max_pr.index(max(max_pr))])

                # Save model only if Sample AUROC is the maximum
                current_avg_score = auroc_sp

                if current_avg_score > best_avg_score:
                    print(f"New best model found at epoch {epoch+1} with Sample Auroc{auroc_sp:.3f}")
                    torch.save({'bn': bn.state_dict(),'decoder': decoder.state_dict()}, ckp_path + str(epoch+1) + str(seed) + 'sample_auc=' + str(auroc_sp) + '.pth')
                    
                    best_avg_score = current_avg_score
                    
                    best_metrics = (auroc_px, auroc_sp, aupro, ap_loc, f1, prec, rec, f1_px)
    return best_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)  # Training epochs
    parser.add_argument('--res', default=3, type=int)  # Select the number of connections, can choose 1, 2, 3, which actually represents 0, 1, 2 connections
    parser.add_argument('--learning_rate', default=0.005, type=float)  # Learning rate
    parser.add_argument('--batch_size', default=16, type=int)  # Batch size
    parser.add_argument('--seed', default=[111,250,444,999,114514], nargs='+', type=int)  # Random seed
    parser.add_argument('--class_', default='all', type=str)  # Select sub-dataset
    parser.add_argument('--seg', default=0, type=int)  # Choose whether segmentation is needed
    parser.add_argument('--print_epoch', default=50, type=int)  # Print every few epochs
    parser.add_argument('--data_path', default='/home/intern24/mvtec/', type=str)  # Path to dataset folder
    parser.add_argument('--save_path', default='/home/intern24/anomaly_checkpoints/dat_train2/skipconnection/', type=str)  # Path to save model files
    parser.add_argument('--print_canshu', default=1, type=int)  # Whether to print anomaly scores for test set
    parser.add_argument('--score_num', default=1, type=int)  # Number of anomaly scores used in the final anomaly score
    parser.add_argument('--print_loss', default=1, type=int)
    parser.add_argument('--img_path', default='/home/intern24/anomaly_checkpoints/dat_train2/skipconnection/result_img/', type=str)  # If segmentation is needed, select the path
    parser.add_argument('--vis', default=0, type=int)  # If segmentation is needed, whether to visualize output
    parser.add_argument('--cut', default=0, type=int)  # Whether to use cutpaste data augmentation
    parser.add_argument('--layerloss', default=1, type=int)  # Whether to use inter-group consistency loss
    parser.add_argument('--rate', default=0.05, type=float)  # Proportion of inter-group consistency loss
    parser.add_argument('--print_max', default=1, type=int)  # Whether to print the best AUC
    parser.add_argument('--net', default='wide_res50', type=str)  # Available net types, can choose res18, res34, res50, wide_res50
    parser.add_argument('--L2', default=0, type=int)  # Whether to use L2 loss function
    parser.add_argument('--contrastive', default=0, type=int)  # Whether to use the dust-vs-defect contrastive loss (branch contrastive_loss)
    parser.add_argument('--contrastive_rate', default=0.1, type=float)  # Weight (lambda) of the contrastive loss term — DA VALIDARE via ablation
    parser.add_argument('--contrastive_temp', default=0.1, type=float)  # Temperature for the N-pair/InfoNCE loss — DA VALIDARE via ablation
    parser.add_argument('--dust_bank_path', default=None, type=str)  # Path to dust_bank/ (contiene raw_images, dust_images, dust_masks). Richiesto se --contrastive 1
    args = parser.parse_args()

    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    if args.class_ == 'all':
        all = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
               'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood', 'reda']
        epoch_ = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        rate_ = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
        for class_, epoch, rate in zip(all, epoch_, rate_):
            print(class_)
            print(epoch)
            print(rate)
            print_epoch = args.print_epoch
            seed = args.seed[0]
            print('*************************')
            print('seed:', seed)
            setup_seed(seed)
            train(class_, epoch, args.learning_rate, args.res, args.batch_size, print_epoch, args.seg, args.data_path, args.save_path, args.print_canshu, args.score_num, args.print_loss, args.img_path, args.vis, args.cut, args.layerloss, rate, args.print_max, args.net, args.L2, seed,
                  contrastive=args.contrastive, contrastive_rate=args.contrastive_rate, contrastive_temp=args.contrastive_temp, dust_bank_path=args.dust_bank_path)
            print('*************************')  

    if args.class_ != 'all':
            for seed in args.seed:
                print('*************************')
                print('seed:', seed)
                setup_seed(seed)
                train(args.class_, args.epochs, args.learning_rate, args.res, args.batch_size, args.print_epoch, args.seg, args.data_path, args.save_path, args.print_canshu, args.score_num, args.print_loss, args.img_path, args.vis, args.cut, args.layerloss, args.rate, args.print_max, args.net, args.L2, seed,
                      contrastive=args.contrastive, contrastive_rate=args.contrastive_rate, contrastive_temp=args.contrastive_temp, dust_bank_path=args.dust_bank_path)
                print('*************************')