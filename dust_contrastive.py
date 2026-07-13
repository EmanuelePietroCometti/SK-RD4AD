"""
dust_contrastive.py
====================
Modulo per la "dust-vs-defect" contrastive loss nel branch `contrastive_loss` di SK-RD4AD.

Perché serve
------------
Il training di RD4AD/SK-RD4AD è puramente ricostruttivo: il decoder studente impara a
ricostruire le feature del teacher (encoder congelato) SOLO sulle immagini "good" viste in
training. Se la polvere non è mai stata vista (o è rara) nel training set, in inferenza il
teacher estrae comunque feature "vere" nella regione di polvere (l'encoder è congelato, non può
ignorarla), ma lo studente non ha mai imparato a ricostruirle -> discrepanza alta -> falso
positivo, anche se la polvere non è semanticamente un difetto.

Questo modulo attacca il problema su due fronti complementari, entrambi radicati in letteratura:

1. Esposizione del decoder alla polvere come "normale" (fix a livello di ricostruzione):
   incollando patch di polvere REALI (dal tuo dust_bank) su immagini pulite e includendo
   anche quella vista nella reconstruction loss standard.

2. Contrastive loss sull'embedding OCBE (`bn(inputs)`, il collo di bottiglia one-class prima
   del decoder) che tira vicino (anchor pulito, view con polvere) e allontana (anchor pulito,
   view con pseudo-difetto sintetico). Combina:
     - CutPaste (Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection and
       Localization", CVPR 2021) per generare pseudo-difetti senza dati anomali etichettati;
     - Multi-class N-pair / InfoNCE loss (Sohn, NeurIPS 2016; Chen et al. "SimCLR", ICML 2020;
       Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020) come obiettivo
       contrastivo, robusto anche con batch piccoli (tipico su Colab, es. batch_size=4) perché
       ogni anchor viene confrontato con TUTTI i negativi del batch, non solo col proprio.

Punto di innesto nel training loop (vedi main.py, blocco "INLINE GPU AUGMENTATION BLOCK"):
    img         # immagine augmentata, in [0,1], CHW*B, PRIMA della renormalize ImageNet
    img_p, img_n = dust_bank.build_views(img)   # positiva (polvere) / negativa (pseudo-difetto)
    ... poi rinormalizzare tutte e tre e passarle all'encoder (concatenate sul batch) ...
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2


class DustBank:
    """
    Indicizza <dust_bank_root>/dust_images/*.bmp e <dust_bank_root>/dust_masks/*.bmp
    (stesso nome file, come da tua struttura) e produce:
      - view "positiva": immagine pulita + patch di polvere REALE incollata (paste_dust)
      - view "negativa": immagine pulita + pseudo-difetto CutPaste/Scar (paste_synthetic_defect)

    NOTA: usa `raw_images/` come riferimento visivo originale (non processato) se un domani
    vuoi ri-generare le maschere; per il training viene usata solo la coppia
    dust_images/dust_masks, già pronta.

    Esegui SEMPRE verify_dust_pipeline.py prima di usare questa classe in un training vero:
    qui assumiamo che i file siano già stati validati (nomi allineati, maschere non vuote).
    """

    def __init__(self, dust_bank_root, device, patch_padding=8, min_mask_pixels=25):
        self.device = device
        self.patch_padding = patch_padding

        img_dir = os.path.join(dust_bank_root, "dust_images")
        mask_dir = os.path.join(dust_bank_root, "dust_masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"Attese le sottocartelle 'dust_images' e 'dust_masks' dentro {dust_bank_root}. "
                f"Esegui verify_dust_pipeline.py per una diagnosi dettagliata."
            )

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.bmp")))
        self.patches = []  # lista di (rgb [3,h,w] float 0..1, alpha [1,h,w] float 0..1)
        skipped = 0

        for img_path in img_paths:
            name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, name)
            if not os.path.exists(mask_path):
                skipped += 1
                continue

            img_bgr = cv2.imread(img_path)
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask_gray is None:
                skipped += 1
                continue

            ys, xs = np.where(mask_gray > 10)
            if len(ys) < min_mask_pixels:
                skipped += 1
                continue

            h, w = mask_gray.shape
            y0, y1 = max(int(ys.min()) - patch_padding, 0), min(int(ys.max()) + patch_padding, h)
            x0, x1 = max(int(xs.min()) - patch_padding, 0), min(int(xs.max()) + patch_padding, w)

            rgb = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            alpha = mask_gray[y0:y1, x0:x1].astype(np.float32) / 255.0
            # feather dei bordi: evita seam netti che il modello potrebbe imparare come
            # "scorciatoia" (stesso problema segnalato per il boundary artifact in DRAEM/NSA)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            alpha_t = torch.from_numpy(alpha).unsqueeze(0).float()
            self.patches.append((rgb_t, alpha_t))

        if len(self.patches) == 0:
            raise RuntimeError(
                f"Nessuna patch di polvere valida trovata in {dust_bank_root}. "
                f"Esegui verify_dust_pipeline.py per capire perché."
            )

        print(f"[DustBank] {len(self.patches)} patch di polvere reale caricate "
              f"({skipped} file scartati per maschera mancante/vuota) da {dust_bank_root}")

    # ------------------------------------------------------------------ #
    # Utility di compositing
    # ------------------------------------------------------------------ #
    def _random_real_patch(self):
        rgb, alpha = random.choice(self.patches)
        if random.random() < 0.5:
            rgb, alpha = torch.flip(rgb, dims=[2]), torch.flip(alpha, dims=[2])
        if random.random() < 0.5:
            rgb, alpha = torch.flip(rgb, dims=[1]), torch.flip(alpha, dims=[1])
        k = random.choice([0, 1, 2, 3])
        if k:
            rgb, alpha = torch.rot90(rgb, k, dims=[1, 2]), torch.rot90(alpha, k, dims=[1, 2])
        return rgb.to(self.device), alpha.to(self.device)

    @staticmethod
    def _alpha_paste(base_img, patch_rgb, patch_alpha, top, left, opacity=1.0):
        """Alpha-compositing di patch_rgb su base_img. Ritorna una NUOVA immagine (no in-place)."""
        _, H, W = base_img.shape
        ph, pw = patch_rgb.shape[-2:]
        ph, pw = min(ph, H - top), min(pw, W - left)
        if ph <= 0 or pw <= 0:
            return base_img.clone()

        patch_rgb = patch_rgb[:, :ph, :pw]
        alpha = (patch_alpha[:, :ph, :pw] * opacity).clamp(0, 1)

        out = base_img.clone()
        region = out[:, top:top + ph, left:left + pw]
        out[:, top:top + ph, left:left + pw] = region * (1 - alpha) + patch_rgb * alpha
        return out

    @staticmethod
    def _rotate(patch, angle_deg):
        return F_v2.rotate(patch.unsqueeze(0), angle_deg, expand=True).squeeze(0)

    # ------------------------------------------------------------------ #
    # View POSITIVA: polvere reale (deve restare vicina all'anchor pulito)
    # ------------------------------------------------------------------ #
    def paste_dust(self, img, opacity_range=(0.55, 1.0)):
        """
        img: tensor [3,H,W] in [0,1] (una SINGOLA immagine, non batchata).
        Ritorna (img_con_polvere [3,H,W], mask_binaria [1,H,W]).
        """
        _, H, W = img.shape
        patch_rgb, patch_alpha = self._random_real_patch()
        ph, pw = patch_rgb.shape[-2:]
        top = random.randint(0, max(H - ph, 0))
        left = random.randint(0, max(W - pw, 0))
        opacity = random.uniform(*opacity_range)

        out = self._alpha_paste(img, patch_rgb, patch_alpha, top, left, opacity)

        # La mask deve rispecchiare l'ESATTA impronta del blending (qualsiasi alpha>0), non una
        # soglia arbitraria: col feathering (Gaussian blur sul bordo) l'alpha decade in modo
        # continuo verso 0, quindi una soglia troppo alta (es. >0.05) lascerebbe fuori pixel che
        # in realtà SONO stati leggermente alterati dal blend, facendo fallire i controlli V&V
        # di tipo "outside-mask must be unchanged".
        mask = torch.zeros((1, H, W), device=img.device)
        eff_h, eff_w = min(ph, H - top), min(pw, W - left)
        if eff_h > 0 and eff_w > 0:
            mask[:, top:top + eff_h, left:left + eff_w] = (
                (patch_alpha[:, :eff_h, :eff_w] * opacity) > 1e-4
            ).float()
        return out.clamp(0, 1), mask

    # ------------------------------------------------------------------ #
    # View NEGATIVA: pseudo-difetto CutPaste / Scar (deve restare lontana dall'anchor)
    # Riferimento: Li, Sohn, Yoon, Pfister — "CutPaste: Self-Supervised Learning for
    # Anomaly Detection and Localization", CVPR 2021.
    # ------------------------------------------------------------------ #
    def paste_synthetic_defect(self, img, donor_img, mode=None):
        """
        img:       [3,H,W] in [0,1], l'immagine su cui incollare il difetto.
        donor_img: [3,H,W] in [0,1], un'ALTRA immagine del batch da cui tagliare la patch
                   (mai la stessa `img`: incollare una texture identica sopra se stessa non
                   crea un segnale utile, il decoder la "spiegherebbe" comunque).
        Ritorna (img_con_difetto [3,H,W], mask_binaria [1,H,W]).
        """
        _, H, W = img.shape
        mode = mode or random.choice(["patch", "scar"])

        if mode == "patch":
            ph = random.randint(max(int(0.05 * H), 4), max(int(0.25 * H), 5))
            pw = random.randint(max(int(0.05 * W), 4), max(int(0.25 * W), 5))
        else:  # "scar": striscia sottile ruotata, utile per graffi/righe/crepe
            ph = random.randint(max(int(0.02 * H), 2), max(int(0.04 * H), 3))
            pw = random.randint(max(int(0.20 * W), 8), max(int(0.50 * W), 9))

        dy = random.randint(0, max(H - ph, 0))
        dx = random.randint(0, max(W - pw, 0))
        patch = donor_img[:, dy:dy + ph, dx:dx + pw].clone()

        # Jitter fotometrico: rompe la scorciatoia "stesso identico pixel = difetto" e forza
        # il modello a ragionare sulla struttura, non sul valore assoluto del pixel.
        patch = patch * random.uniform(0.7, 1.3) + random.uniform(-0.05, 0.05)
        patch = patch.clamp(0, 1)

        if mode == "scar":
            angle = random.uniform(-45, 45)
            patch = self._rotate(patch, angle)

        ph, pw = patch.shape[-2:]
        top = random.randint(0, max(H - ph, 0))
        left = random.randint(0, max(W - pw, 0))
        alpha = torch.ones((1, ph, pw), device=img.device)

        out = self._alpha_paste(img, patch, alpha, top, left, opacity=1.0)

        mask = torch.zeros((1, H, W), device=img.device)
        eff_h, eff_w = min(ph, H - top), min(pw, W - left)
        mask[:, top:top + eff_h, left:left + eff_w] = 1.0
        return out.clamp(0, 1), mask

    # ------------------------------------------------------------------ #
    # Wrapper batched, da chiamare direttamente in main.py
    # ------------------------------------------------------------------ #
    def build_views(self, img_batch):
        """
        img_batch: [B,3,H,W] in [0,1] (già croppato/augmentato, PRIMA della renormalize).
        Ritorna (img_pos [B,3,H,W], img_neg [B,3,H,W]).

        Il "donor" per i pseudo-difetti è lo stesso batch shiftato di 1 posizione (roll):
        garantisce che il donor sia sempre un'immagine diversa dall'anchor corrente, senza
        bisogno di tenere in memoria un secondo dataloader.
        """
        B = img_batch.shape[0]
        if B < 2:
            raise ValueError(
                "La contrastive loss richiede batch_size >= 2 (serve un'immagine 'donor' "
                "diversa da se stessa per generare i pseudo-difetti). Con batch_size=1 "
                "disattiva --contrastive."
            )
        donors = img_batch.roll(shifts=1, dims=0)

        pos_list, neg_list = [], []
        for b in range(B):
            p_img, _ = self.paste_dust(img_batch[b])
            n_img, _ = self.paste_synthetic_defect(img_batch[b], donors[b])
            pos_list.append(p_img)
            neg_list.append(n_img)
        return torch.stack(pos_list, dim=0), torch.stack(neg_list, dim=0)


class ProjectionHead(nn.Module):
    """
    Head di proiezione stile SimCLR/SupCon (MLP a 2 layer + L2-norm) applicata all'embedding
    OCBE (`bn(inputs)`), tipicamente [B, 2048, 8, 8] per wide_res50/res50 a risoluzione 256
    (2048 = 512 * block.expansion, expansion=4 per i Bottleneck block; per res18/res34,
    che usano BasicBlock con expansion=1, sarà invece [B, 512, 8, 8]).

    `in_channels` va dedotto a runtime (vedi main.py: una forward "a vuoto" prima di costruire
    l'head) invece di hardcodarlo, per non rompersi silenziosamente cambiando --net.
    """

    def __init__(self, in_channels, hidden_dim=512, out_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.net(x)
        return F.normalize(x, dim=1)


def dust_contrastive_loss(z_anchor, z_pos, z_neg, temperature=0.1):
    """
    Multi-class N-pair / InfoNCE loss (Sohn, NeurIPS 2016; SimCLR, Chen et al. ICML 2020;
    SupCon, Khosla et al. NeurIPS 2020).

      z_anchor : embedding immagine pulita augmentata          [B, D] (L2-normalizzati)
      z_pos    : embedding stessa immagine + polvere reale      [B, D]  -> deve restare vicino
      z_neg    : embedding stessa immagine + pseudo-difetto     [B, D]  -> deve restare lontano

    Per ogni anchor i, i negativi usati sono TUTTI gli z_neg del batch (non solo z_neg[i]):
    con batch piccoli (es. batch_size=4 su Colab) questo dà comunque B confronti negativi per
    anchor invece di 1 solo, a differenza di SimCLR "puro" che ha bisogno di batch grandi per
    essere efficace.
    """
    assert z_anchor.shape == z_pos.shape == z_neg.shape, "z_anchor/z_pos/z_neg devono avere la stessa shape [B, D]"

    sim_pos = (z_anchor * z_pos).sum(dim=1) / temperature          # [B]
    sim_neg = (z_anchor @ z_neg.t()) / temperature                 # [B, B]

    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)     # [B, 1+B]
    labels = torch.zeros(z_anchor.shape[0], dtype=torch.long, device=z_anchor.device)

    return F.cross_entropy(logits, labels)
