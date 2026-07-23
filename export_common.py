"""
export_common.py — Contratto di export ONNX condiviso dalle 4 architetture.

Unica fonte di verita' per: nomi I/O, opset, assi dinamici, schema metadati,
normalizzazione in-graph, verifica di parita', guard sullo stato addestrato.
Nessun exporter deve reimplementare questi elementi: ogni duplicazione e' una
divergenza futura.

Questo file va copiato IDENTICO (byte per byte) nei tre repository:
    SuperSimpleNet/export_common.py
    anomaly_detection_for_textile_industry/src/export_common.py
    SK-RD4AD/export_common.py
Se lo modifichi in uno, propagalo negli altri e riscrivi EXPORT_COMMON_SHA256
(vedi check_contract_sync.py, che fallisce se le copie divergono o se l'hash
non corrisponde al contenuto).

Il perimetro normativo e' la sezione 2 di CONTRACT.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Hash di integrita' del contratto: sha256 esadecimale di QUESTO file con il
# valore di EXPORT_COMMON_SHA256 sostituito da 64 '0'. Serve a due cose:
# (1) rilevare se una copia e' stata modificata senza aggiornare l'hash;
# (2) dare a check_contract_sync.py un valore atteso stabile da confrontare tra
# le tre repo. Rigeneralo con:  python check_contract_sync.py --write-hash
EXPORT_COMMON_SHA256 = "ba2a1cef8c785c63596ba9ad4756d73d487795f802e59dea75f856b054219587"

EXPORT_CONTRACT = "3.0"
OPSET = 17

INPUT_NAME = "image"
OUTPUT_NAMES = ["anomaly_map", "anomaly_score"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DYNAMIC_AXES = {
    INPUT_NAME: {0: "batch"},
    "anomaly_map": {0: "batch"},
    "anomaly_score": {0: "batch"},
}


class InGraphNormalize(nn.Module):
    """Normalizzazione ImageNet come nodo del grafo.

    L'input del grafo e' RGB in [0,1]; questo modulo applica (x - mean) / std.
    Tenerla qui invece che sull'host elimina la classe di bug
    "normalizzazione dimenticata / applicata due volte", che ha gia' colpito
    SuperSimpleNet/eval.py (AUROC 0.91 -> 0.51) e inference-gpu.cpp.

    I buffer sono registrati (non costanti Python) cosi' finiscono come
    initializer del grafo e sono ispezionabili con netron.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ExportWrapper(nn.Module):
    """Base comune agli export delle 4 architetture.

    Contratto delle sottoclassi: implementare `core(x_normalized)` che
    restituisce (anomaly_map_finale, anomaly_score_finale). Blur, sigmoid e
    riduzione a scalare vanno DENTRO `core`: il runtime non fa alcun
    post-processing decisionale.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.normalize = InGraphNormalize(mean, std)

    def core(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, image: torch.Tensor):
        anomaly_map, anomaly_score = self.core(self.normalize(image))
        # Forma canonica: mappa [B,1,H,W], score [B]. Uniformarla qui evita
        # che ogni architettura la interpreti a modo suo nel runtime C++.
        if anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)
        anomaly_score = anomaly_score.reshape(anomaly_score.shape[0])
        return anomaly_map, anomaly_score


def assert_trained_bn(module: nn.Module, context: str) -> None:
    """Rifiuta l'export se i BatchNorm di `module` non hanno mai visto dati.

    Un BatchNorm appena inizializzato ha `num_batches_tracked == 0`; dopo il
    training (o dopo aver caricato uno state_dict addestrato, che include quel
    buffer) e' > 0. E' un sentinello di stato addestrato affidabile per le
    teste allenate di SuperSimpleNet e per il decoder di SK-RD4AD.

    ATTENZIONE: passare SOLO il sotto-modulo effettivamente addestrato. Un
    backbone pretrained (feature_extractor) porta num_batches_tracked ereditati
    da ImageNet e falserebbe il controllo.

    Solleva RuntimeError con l'indicazione di cosa manca; non fa mai fallback.
    """
    counts = [
        int(m.num_batches_tracked)
        for m in module.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        and m.num_batches_tracked is not None
    ]
    if not counts:
        raise RuntimeError(
            f"{context}: nessun BatchNorm con num_batches_tracked trovato in "
            f"'{module.__class__.__name__}'. Il guard sullo stato addestrato "
            "non e' applicabile: verifica di aver passato il sotto-modulo "
            "allenato giusto, o aggiungi un sentinello specifico per questa "
            "architettura (cerca i buffer inizializzati a un valore noto)."
        )
    if all(c == 0 for c in counts):
        raise RuntimeError(
            f"{context}: tutti i BatchNorm hanno num_batches_tracked == 0 → il "
            "modello non e' addestrato (buffer ai valori di init). Esporto un "
            "grafo valido ma numericamente privo di senso. Carica un checkpoint "
            "addestrato, oppure — se vuoi solo testare la pipeline di export — "
            "usa esplicitamente --self_test (che marca weights_source="
            "random_self_test, rifiutato dal runtime)."
        )


def sha256_of(path: Path | None) -> str:
    if path is None:
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(*, architecture: str, image_size: tuple[int, int],
                   blur_kernel_size: int, blur_sigma: float,
                   dynamic_crop: bool = False, weights_path: Path | None = None,
                   resize_mode: str = "bilinear_antialias",
                   extra: dict | None = None) -> dict[str, str]:
    """Metadati nello schema del contratto 3.0.

    Tutti i valori sono stringhe: e' l'unico tipo ammesso da
    onnx.ModelProto.metadata_props, e il runtime C++ li converte esplicitamente.
    """
    meta = {
        "export_contract": EXPORT_CONTRACT,
        "architecture": architecture,
        "input_layout": "NCHW_RGB_0_1",
        "input_height": str(image_size[0]),
        "input_width": str(image_size[1]),
        "normalization": "in_graph",
        "norm_mean": ",".join(f"{v:.6f}" for v in IMAGENET_MEAN),
        "norm_std": ",".join(f"{v:.6f}" for v in IMAGENET_STD),
        "resize_mode": resize_mode,
        "dynamic_crop": "true" if dynamic_crop else "false",
        "score_semantics": "final",
        "map_semantics": "final_blurred",
        "blur_kernel_size": str(blur_kernel_size),
        "blur_sigma": f"{blur_sigma:.4f}",
        "weights_source": (f"checkpoint:{weights_path.name}"
                           if weights_path is not None else "random_self_test"),
        "weights_sha256": sha256_of(weights_path),
        # verified/calibrated_threshold/calibration_global_* sono scritti dopo,
        # da calibrate_threshold.py: non possono essere noti al momento
        # dell'export perche' dipendono da un run sul dataset etichettato.
        "verified": "false",
    }
    if extra:
        meta.update({k: str(v) for k, v in extra.items()})
    return meta


def export(wrapper: nn.Module, image_size: tuple[int, int], onnx_path: Path,
           device: str, metadata: dict[str, str]) -> Path:
    """Export fp32 con assi dinamici + scrittura metadati.

    batch=2 nel tensore di traccia: con batch=1 l'exporter puo' specializzare
    silenziosamente una dimensione anche in presenza di dynamic_axes.
    """
    import onnx

    h, w = image_size
    # Input in [0,1]: usare rand (non randn) fa si' che la traccia veda lo
    # stesso dominio dei dati reali; conta per do_constant_folding e per
    # eventuali clamp.
    dummy = torch.rand(2, 3, h, w, dtype=torch.float32, device=device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
            input_names=[INPUT_NAME], output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,        # l'exporter classico rispetta dynamic_axes
            external_data=False, # singolo file .onnx autoconsistente
        )

    onnx.checker.check_model(str(onnx_path))
    model = onnx.load(str(onnx_path))
    for k, v in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(model, str(onnx_path))
    return onnx_path


def read_metadata(onnx_path: Path) -> dict[str, str]:
    """Legge metadata_props dal file .onnx come dict."""
    import onnx
    model = onnx.load(str(onnx_path))
    return {p.key: p.value for p in model.metadata_props}


def update_metadata(onnx_path: Path, updates: dict[str, str]) -> None:
    """Aggiorna (o aggiunge) chiavi nei metadata_props del file .onnx.

    Le chiavi esistenti vengono sostituite, non duplicate: una ri-calibrazione
    deve sovrascrivere calibrated_threshold/verified, non accumularne copie
    (onnx conserva l'ordine e un lettore ingenuo prenderebbe la prima).
    """
    import onnx
    model = onnx.load(str(onnx_path))
    existing = {p.key: p for p in model.metadata_props}
    for k, v in updates.items():
        if k in existing:
            existing[k].value = str(v)
        else:
            entry = model.metadata_props.add()
            entry.key, entry.value = k, str(v)
    onnx.save(model, str(onnx_path))


def verify(wrapper: nn.Module, onnx_path: Path, image_size: tuple[int, int],
           device: str, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    """Parita' PyTorch <-> ONNX Runtime su input in [0,1].

    Tolleranze: PyTorch e ORT usano kernel di convoluzione e ordini di
    riduzione diversi, quindi la parita' bit-exact e' impossibile su reti
    profonde. 1e-3 intercetta i bug veri (op sbagliata, pesi trasposti, blur
    mancante) lasciando passare la normale deriva in virgola mobile.

    Si testano batch 1 e 4 per dimostrare che l'asse dinamico funziona: un
    grafo specializzato a batch=2 passerebbe un test a solo batch=2.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    h, w = image_size

    print(f"\n--- PyTorch vs ONNX parity (atol={atol:g}, rtol={rtol:g}) ---")
    for batch in (1, 4):
        x = rng.random((batch, 3, h, w), dtype=np.float32)
        with torch.no_grad():
            tm, ts = wrapper(torch.from_numpy(x).to(device))
        om, osc = sess.run(OUTPUT_NAMES, {INPUT_NAME: x})
        tm, ts = tm.cpu().numpy(), ts.cpu().numpy()
        np.testing.assert_allclose(om, tm, atol=atol, rtol=rtol,
                                   err_msg=f"anomaly_map mismatch (batch={batch})")
        np.testing.assert_allclose(osc, ts, atol=atol, rtol=rtol,
                                   err_msg=f"anomaly_score mismatch (batch={batch})")
        print(f"  batch={batch}: map |d|max={np.abs(om-tm).max():.2e}  "
              f"score |d|max={np.abs(osc-ts).max():.2e}  OK")
    print("[PASS] parita' numerica entro tolleranza.")


def resolve_output_path(output: str | None, weights: Path | None,
                        default_stem: str) -> Path:
    """`output` puo' essere un file .onnx o una directory.

    Una directory e' riconosciuta se esiste come tale o se manca il suffisso
    .onnx; in quel caso il nome file deriva dal checkpoint. Evita
    l'IsADirectoryError di torch.onnx.export quando gli si passa una cartella.
    """
    stem = weights.stem if weights is not None else default_stem
    if output:
        out = Path(output)
        if out.is_dir() or out.suffix.lower() != ".onnx":
            out = out / f"{stem}.onnx"
    elif weights is not None:
        out = weights.with_suffix(".onnx")
    else:
        out = Path(f"{stem}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
