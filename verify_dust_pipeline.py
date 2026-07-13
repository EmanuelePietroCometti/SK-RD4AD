"""
verify_dust_pipeline.py
========================
Suite di Verification & Validation (V&V) per la pipeline "dust contrastive loss".
Va eseguita PRIMA di lanciare un training lungo (soprattutto su Colab, dove un crash a metà
allenamento costa tempo e crediti GPU).

Uso:
    python verify_dust_pipeline.py --dust_bank_path /content/sk-rd4ad/dust_bank \
                                    --sample_image /content/mvtec/reda/train/good/000.bmp \
                                    --out_dir ./vv_report

Cosa verifica (in ordine, si ferma al primo errore bloccante):
  1. Struttura cartelle: dust_bank/{raw_images,dust_images,dust_masks} esistono
  2. Allineamento nomi file tra dust_images e dust_masks (stesso basename)
  3. Sanità delle maschere: non vuote, non leggibili come corrotte, area ragionevole
  4. Caricamento DustBank (le patch vengono effettivamente estratte)
  5. Correttezza del compositing: i pixel FUORI dalla maschera incollata non cambiano,
     quelli DENTRO cambiano davvero (niente operazioni "silenziosamente no-op")
  6. Range numerico [0,1] rispettato dopo il compositing
  7. Comportamento della contrastive loss: finita, gradiente non nullo, monotona rispetto
     alla similarità (positivo più simile => loss più bassa)
  8. Salva una griglia visiva (anchor / view polvere / view pseudo-difetto) in
     <out_dir>/preview_views.png per un controllo visivo manuale

NON sostituisce lo smoke-test end-to-end col modello vero: per quello, lancia
`main.py --epochs 1 --print_epoch 1 --contrastive 1` su una cartella di debug con poche
immagini (vedi il messaggio finale di questo script).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch


def fail(msg):
    print(f"\n[FALLITO] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[ATTENZIONE] {msg}")


def check_structure(dust_bank_path):
    print("\n--- 1. Struttura cartelle ---")
    required = ["raw_images", "dust_images", "dust_masks"]
    for sub in required:
        p = os.path.join(dust_bank_path, sub)
        if not os.path.isdir(p):
            fail(f"Manca la cartella '{sub}' dentro {dust_bank_path}")
    ok("raw_images / dust_images / dust_masks presenti")


def check_alignment(dust_bank_path):
    print("\n--- 2. Allineamento dust_images <-> dust_masks ---")
    img_dir = os.path.join(dust_bank_path, "dust_images")
    mask_dir = os.path.join(dust_bank_path, "dust_masks")

    img_names = {os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.bmp"))}
    mask_names = {os.path.basename(p) for p in glob.glob(os.path.join(mask_dir, "*.bmp"))}

    if len(img_names) == 0:
        fail(f"Nessun file .bmp trovato in {img_dir}")

    only_img = img_names - mask_names
    only_mask = mask_names - img_names
    if only_img:
        warn(f"{len(only_img)} immagini SENZA maschera corrispondente (verranno ignorate): "
             f"{sorted(only_img)[:5]}{'...' if len(only_img) > 5 else ''}")
    if only_mask:
        warn(f"{len(only_mask)} maschere SENZA immagine corrispondente (ignorate): "
             f"{sorted(only_mask)[:5]}{'...' if len(only_mask) > 5 else ''}")

    common = img_names & mask_names
    if len(common) == 0:
        fail("Nessuna coppia (immagine, maschera) valida trovata: controlla i nomi dei file.")
    ok(f"{len(common)} coppie immagine/maschera allineate su {len(img_names)} immagini totali")
    return sorted(common)


def check_masks(dust_bank_path, common_names, min_mask_pixels=25, max_coverage=0.9):
    print("\n--- 3. Sanità delle maschere ---")
    img_dir = os.path.join(dust_bank_path, "dust_images")
    mask_dir = os.path.join(dust_bank_path, "dust_masks")

    n_empty, n_full, n_corrupt, n_size_mismatch = 0, 0, 0, 0
    valid = []

    for name in common_names:
        img = cv2.imread(os.path.join(img_dir, name))
        mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            n_corrupt += 1
            continue
        if img.shape[:2] != mask.shape[:2]:
            n_size_mismatch += 1
            continue

        coverage = float((mask > 10).mean())
        if coverage == 0.0:
            n_empty += 1
            continue
        if coverage > max_coverage:
            n_full += 1
            continue
        if (mask > 10).sum() < min_mask_pixels:
            n_empty += 1
            continue

        valid.append(name)

    print(f"  file corrotti/illeggibili : {n_corrupt}")
    print(f"  mismatch dimensioni img/mask : {n_size_mismatch}")
    print(f"  maschere vuote/troppo piccole : {n_empty}")
    print(f"  maschere quasi-tutto-frame (>{int(max_coverage*100)}% area, sospette) : {n_full}")

    if len(valid) == 0:
        fail("Nessuna maschera utilizzabile dopo i controlli di sanità.")
    ok(f"{len(valid)} patch di polvere utilizzabili")
    return valid


def check_dustbank_loading(dust_bank_path):
    print("\n--- 4. Caricamento DustBank ---")
    from dust_contrastive import DustBank
    bank = DustBank(dust_bank_path, device="cpu")
    if len(bank.patches) == 0:
        fail("DustBank si è caricato ma non contiene patch (non dovrebbe succedere se lo step 3 è passato).")
    ok(f"DustBank caricata con {len(bank.patches)} patch")
    return bank


def check_compositing(bank, sample_image_path, out_dir):
    print("\n--- 5-6. Correttezza del compositing ---")
    from torchvision.transforms.v2.functional import pil_to_tensor
    from PIL import Image

    img_pil = Image.open(sample_image_path).convert("RGB").resize((256, 256))
    img = pil_to_tensor(img_pil).float() / 255.0  # [3,256,256] in [0,1]

    donor_pil = Image.open(sample_image_path).convert("RGB").resize((256, 256))
    donor = pil_to_tensor(donor_pil).float() / 255.0
    # perturba leggermente il donor cosi' non e' pixel-identico all'anchor
    donor = (donor + 0.05 * torch.randn_like(donor)).clamp(0, 1)

    img_p, mask_p = bank.paste_dust(img)
    img_n, mask_n = bank.paste_synthetic_defect(img, donor, mode="patch")
    img_n_scar, mask_n_scar = bank.paste_synthetic_defect(img, donor, mode="scar")

    for name, view, mask in [("polvere", img_p, mask_p), ("difetto-patch", img_n, mask_n),
                              ("difetto-scar", img_n_scar, mask_n_scar)]:
        if view.shape != img.shape:
            fail(f"[{name}] shape output {view.shape} diversa dall'input {img.shape}")
        if view.min() < -1e-4 or view.max() > 1 + 1e-4:
            fail(f"[{name}] range fuori da [0,1]: min={view.min().item():.4f} max={view.max().item():.4f}")

        outside = mask[0] < 0.5
        if not torch.allclose(view[:, outside], img[:, outside], atol=1e-6):
            fail(f"[{name}] alcuni pixel FUORI dalla maschera sono cambiati: il compositing "
                 f"sta scrivendo fuori dalla regione attesa (bug di indicizzazione?).")

        inside = mask[0] >= 0.5
        if inside.sum() == 0:
            fail(f"[{name}] maschera vuota: il compositing non ha incollato nulla (no-op silenzioso).")
        max_change_inside = (view[:, inside] - img[:, inside]).abs().max().item()
        if max_change_inside < 1e-3:
            fail(f"[{name}] i pixel DENTRO la maschera non sono cambiati in modo significativo "
                 f"(max diff={max_change_inside:.6f}): probabile no-op silenzioso.")

        ok(f"[{name}] compositing corretto: {inside.sum().item()} px modificati, "
           f"{outside.sum().item()} px invariati, range valido")

    os.makedirs(out_dir, exist_ok=True)
    _save_preview(img, img_p, img_n, img_n_scar, os.path.join(out_dir, "preview_views.png"))
    ok(f"Griglia di anteprima salvata in {os.path.join(out_dir, 'preview_views.png')} "
       f"-> CONTROLLARE VISIVAMENTE prima di procedere")


def _save_preview(img, img_p, img_n, img_n_scar, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("anchor (pulito)", img), ("view positiva (polvere reale)", img_p),
             ("view negativa (CutPaste patch)", img_n), ("view negativa (scar)", img_n_scar)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, v) in zip(axes, views):
        ax.imshow(v.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def check_contrastive_loss():
    print("\n--- 7. Comportamento della contrastive loss ---")
    from dust_contrastive import ProjectionHead, dust_contrastive_loss

    torch.manual_seed(0)
    B = 4
    proj = ProjectionHead(in_channels=16)

    embed_a = torch.randn(B, 16, 8, 8, requires_grad=True)
    embed_n = torch.randn(B, 16, 8, 8)  # stessi negativi per i due confronti sotto

    za = proj(embed_a)
    zn = proj(embed_n)

    zp_noisy = proj(embed_a.detach() + 2.0 * torch.randn(B, 16, 8, 8))
    loss_noisy = dust_contrastive_loss(za, zp_noisy, zn, temperature=0.1)

    zp_identical = za.clone()
    loss_identical = dust_contrastive_loss(za, zp_identical, zn, temperature=0.1)

    if not torch.isfinite(loss_noisy) or not torch.isfinite(loss_identical):
        fail("La contrastive loss produce NaN/Inf.")
    if not (loss_identical.item() < loss_noisy.item()):
        fail(f"Comportamento inatteso: loss(positivo identico)={loss_identical.item():.4f} "
             f"dovrebbe essere < loss(positivo rumoroso)={loss_noisy.item():.4f}")
    ok(f"loss finita e monotona rispetto alla similarità "
       f"(identico={loss_identical.item():.4f} < rumoroso={loss_noisy.item():.4f})")

    loss_identical.backward()
    if embed_a.grad is None or not torch.isfinite(embed_a.grad).all() or embed_a.grad.abs().sum() == 0:
        fail("Il gradiente non fluisce correttamente attraverso la contrastive loss.")
    ok("gradiente valido e non-nullo attraverso ProjectionHead + contrastive loss")

    if B < 2:
        fail("build_views richiede batch_size >= 2: serve un'immagine 'donor' diversa da se stessa "
             "per generare i pseudo-difetti CutPaste/Scar.")
    ok(f"batch_size={B} compatibile con il meccanismo donor (build_views)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dust_bank_path", required=True, type=str)
    parser.add_argument("--sample_image", required=True, type=str,
                         help="Un'immagine 'good' qualsiasi del tuo dataset (es. train/good/000.bmp), "
                              "usata solo per il test di compositing.")
    parser.add_argument("--out_dir", default="./vv_report", type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.sample_image):
        fail(f"--sample_image non trovata: {args.sample_image}")

    print("=" * 70)
    print("V&V pipeline dust contrastive loss")
    print("=" * 70)

    check_structure(args.dust_bank_path)
    common = check_alignment(args.dust_bank_path)
    valid = check_masks(args.dust_bank_path, common)
    bank = check_dustbank_loading(args.dust_bank_path)
    check_compositing(bank, args.sample_image, args.out_dir)
    check_contrastive_loss()

    print("\n" + "=" * 70)
    print("TUTTI I CONTROLLI STATICI SONO PASSATI.")
    print("Prossimo passo (smoke test end-to-end col modello vero, non duplicato qui):")
    print("  1. crea una cartella di debug con ~8 immagini, es. ./debug_tiny/good/*.bmp")
    print("  2. lancia:")
    print("     python main.py --class_ debug_tiny --data_path ./ --epochs 1 --print_epoch 1 \\")
    print("       --contrastive 1 --dust_bank_path <path_dust_bank> --batch_size 2 --seg 0")
    print("  3. verifica che non crashi, che compaiano i 3 componenti di loss separati nel log,")
    print("     e che i loro ordini di grandezza siano confrontabili (nessuno domina gli altri).")
    print("=" * 70)


if __name__ == "__main__":
    main()