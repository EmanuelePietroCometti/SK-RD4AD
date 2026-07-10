# Skip-Connected Reverse Distillation for Robust One-Class Anomaly Detection

[![CVPR 2025 Workshop](https://img.shields.io/badge/Presented%20at-CVPRW%202025-orange)](https://sites.google.com/view/vand30cvpr2025)
[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Park_SK-RD4AD__Skip-Connected_Reverse_Distillation_For_Robust_One-Class_Anomaly_Detection_CVPRW_2025_paper.html)
> 📣 Accepted at **CVPR 2025 Workshop (VAND 3.0)**  
> 🔧 Official PyTorch implementation of our paper, SK-RD4AD.

## 📖 Introduction

SK-RD4AD (Skip-Connected Reverse Distillation for Anomaly Detection) introduces a novel and effective architecture for one-class anomaly detection. By leveraging non-corresponding skip connections within a reverse knowledge distillation framework, SK-RD4AD effectively mitigates deep feature degradation, a common issue in traditional distillation-based methods. This enhancement significantly improves both pixel-level anomaly localization and image-level detection across various industrial domains.

## 🔥 Key Highlights
- 🔗 **Non-Corresponding Skip Connections**  
  Enhances multi-scale feature propagation between encoder and decoder, allowing both low-level textures and high-level semantics to be preserved.

- 🚀 **State-of-the-Art Performance**  
  Surpasses RD4AD on **MVTec-AD**, **VisA**, and **VAD** by up to **+3.5% AUROC**, **+21% AUPRO**, and excels in challenging categories like **Transistor**.

- 🌐 **Generalization Power**  
  Effectively handles diverse anomaly types and generalizes to **real-world datasets** such as **automotive VAD** and **industrial VisA**.

- ⚙️ **Efficient Architecture**  
  Lightweight decoder with modest memory (401MB) and inference time (0.37s), making it suitable for **edge and cloud** deployment.

---

## 📐 Canonical inference pipeline (training ↔ ONNX parity)

Every score-producing path in this repo — training-time model selection
(`test.py`), the reference evaluation (`eval.py`), the ONNX export
(`export_onnx_from_checkpoint.py`, contract 2.0) and the production runtime —
implements **one** definition:

```
resize 256 → scale to [0,1] → dynamic crop (bg 0.94, pad 30) → ImageNet normalize
→ sum of per-layer (1 − cos similarity) maps, bilinear align_corners=False
→ Gaussian blur k=15 σ=4 zero-padding   (baked INTO the ONNX graph)
→ image score = max of the BLURRED map  (baked INTO the ONNX graph)
```

The single source of truth is `test.py` (`GAUSS_KERNEL_SIZE`, `GAUSS_SIGMA`,
`get_gaussian_kernel`, `compute_anomaly_map_torch`) — never redefine the kernel
or the map elsewhere. Workflow for shipping a model:

1. `python export_onnx_from_checkpoint.py ckpt.pth out.onnx --res <res>` — exports + verifies graph parity.
2. `python parity_check.py --checkpoint ckpt.pth --model out.onnx --data_path ... --class_ ...` — PyTorch↔ONNX on real images (must PASS).
3. `python calibrate_threshold.py --model out.onnx --data_path ... --class_ ... --embed` — computes the F1-optimal threshold **through the ONNX pipeline** and embeds it in the model metadata (`calibrated_threshold`).
4. Cross-check with `eval.py`'s `calibration_pytorch.json`: the two thresholds must agree to floating-point precision.

---


## 📂 Model Overview
SK-RD4AD enhances the original RD4AD framework by addressing deep-layer information loss. It introduces strategically designed non-corresponding skip connections that allow features from shallower teacher layers to influence deeper student layers. This architecture improves the decoder’s capacity to reconstruct complex features and leads to significantly better anomaly localization performance.


![image](https://github.com/user-attachments/assets/052d2923-5fa7-45ed-80e1-00ac0e01efd9)


## ⚙️ Experiment Settings

### 🧪 Installation
```bash
pip install -r requirements.txt
```

### 📁 Dataset Preparation
Download the MVTec, VAD and VisA datasets from:
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [Valeo Anomaly Dataset (VAD)](https://drive.google.com/file/d/1LbHHJHCdkvhzVqekAIRdWjBWaBHxPjuu/view/)
- [visA Dataset](https://github.com/amazon-science/spot-diff)

### 🏃 Train and Test the Model
To train and evaluate the model, use the appropriate script depending on the dataset.

The following command is for training and evaluating the model on **MVTec-AD** :
```bash
python main.py \
    --epochs 200 \
    --res 3 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --seed 111 \
    --class all \
    --seg 1 \
    --print_epoch 10 \
    --data_path /home/user/mvtec/ \
    --save_path /home/user/anomaly_checkpoints/skipconnection/ \
    --print_loss 1 \
    --net wide_res50 \
    --L2 0
```
> For **VisA** or **VAD**, use `main_visa.py` or `main_vad.py`.

## 📈 Performance Highlights
The model is evaluated using AUROC and AUPRO metrics at pixel level. SK-RD4AD consistently shows significant improvements over RD4AD across MVTec-AD, Valeo VAD, and VisA datasets, especially in categories requiring fine-grained spatial reasoning. : 

### MVTec-AD Dataset Performance
| Category     | RD4AD (Pixel AUROC/AUPRO) | SK-RD4AD (Pixel AUROC/AUPRO)   |
|--------------|----------------------------|--------------------------------|
| Carpet       | 98.9 / 97.0                | **99.2 / 97.7**                |
| Bottle       | 98.7 / 96.6                | **98.8 / 96.9**                |
| Hazelnut     | 98.9 / 95.5                | **99.1 / 96.2**                |
| Leather      | 99.4 / 99.1                | **99.6 / 99.2**                |
| **Total Avg**| 97.8 / 93.9                | **98.06 / 94.69**              |

### Valeo Anomaly Dataset (VAD) Performance
| Setting   | Baseline AUROC | SK-RD4AD AUROC |
|-----------|----------------|----------------|
| One-Class | 84.5           | **87.0**       |

### VisA Dataset Performance
| Category     | RD4AD (Pixel AUROC / AUPRO) | SK-RD4AD (Pixel AUROC / AUPRO) |
|--------------|-----------------------------|--------------------------------|
| PCB1         | 99.6 / 43.2                 | 99.6 / **93.7**                |
| PCB2         | 98.3 / 46.4                 | 98.3 / **89.2**                |
| PCB3         | 99.3 / 80.3                 | 98.3 / **90.3**                |
| PCB4         | 98.2 / 72.2                 | **98.6 / 89.0**                |
| Pipe Fryum   | 99.1 / 68.3                 | 99.1 / **94.8**                |
| **Total Avg** | 97.8 / 70.9                 | **98.5 / 92.1**                |

### Model Complexity

| Model       | Time (s) | Memory (MB) | AUROC |
|-------------|----------|-------------|--------|
| RD4AD       | 0.31     | 352         | 97.3   |
| **SK-RD4AD** | 0.37     | 401         | **98.06** |

> Slight increase in cost yields **substantial performance boost**.

## 🖼️ Visualization
<p align="center">
  <img src="https://github.com/user-attachments/assets/b2fe4e4b-6a4c-4c86-8caa-ebef8da92dd8" alt="figg3" width="45%">
  <img src="https://github.com/user-attachments/assets/dbbd9d8a-f70a-4a8f-9a9b-49e2f95ed4be" alt="ffig2" width="45%">

</p>

The visualization results demonstrate **the effectiveness of the SK-RD4AD model in detecting anomalies**. The anomaly maps highlight areas where the model identifies potential defects, using red and yellow hues to indicate regions of high confidence. The overlaid images combine the original images with the anomaly maps, clearly showing the detected anomalies' locations.

## 🎯 Conclusion
SK-RD4AD addresses the challenge of deep feature loss in anomaly detection by introducing **non-corresponding skip connections** within a reverse distillation framework.  
This design improves **multi-scale feature retention** and enhances the model's ability to detect **subtle and diverse anomalies**.  
Tested across industrial benchmarks, it shows consistent improvements over RD4AD, making it a **robust and effective solution for real-world anomaly detection tasks**.

## 📚 Citation

```
@InProceedings{Park_2025_CVPR,
    author    = {Park, EunJu and Kim, Taekyung and Kim, Minju and Lee, Hojun and Lee, Gil-Jun},
    title     = {SK-RD4AD : Skip-Connected Reverse Distillation For Robust One-Class Anomaly Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {3945-3953}
}
```
## 🙏 Acknowledgement

This work builds upon the [RD4AD (Anomaly Detection via Reverse Distillation From One-Class Embedding)](https://github.com/hq-deng/RD4AD) framework.  
We sincerely thank the original authors for open-sourcing their code and pre-trained models, which made this research possible.



