"""Optuna HPO per SK-RD4AD, in due stadi (obiettivi: AUROC massimo, parametri minimi).

Da lanciare dalla root della repo SK-RD4AD (accanto a main.py):

    python hpo_skrd4ad.py --stage screen --category carpet --datasets_folder ./mvtec/   # stadio A
    python hpo_skrd4ad.py --stage tune   --category carpet --datasets_folder ./mvtec/   # stadio B
    (--stage all: A poi B;  --stage report: solo tabelle)

Struttura identica a hpo_ssn.py: la sezione ADAPTER (specifica della repo) e' l'unica diversa,
il blocco COMMON e' lo stesso in entrambi i file.
  * stadio A: 12 architetture (net x res) con iperparametri di default, senza pruning -> fronte
    di Pareto (AUROC max, parametri min);
  * stadio B: uno studio per architettura del fronte, TPE + MedianPruner, 200 epoche per trial;
  * AUROC = AUROC di immagine (sample) del checkpoint selezionato da main.py (regola della repo, NON
    modificata); augmentation disattivata (AugConfig() = tutto off).
"""
import gc  # noqa: F401  (usato dal blocco COMMON)
import multiprocessing  # noqa: F401
import shutil
from pathlib import Path

import optuna
import torch
from optuna.distributions import IntDistribution

from aug_config import AugConfig
import main as skrd_main
from main import setup_seed

# ======================================================================================
# ADAPTER -- specifico di SK-RD4AD
# ======================================================================================
NAME = "skrd4ad"
DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 32  # default di main.py

# Architetture dal piu' piccolo al piu' grande (net-major)
NETS = ["res18", "res34", "res50", "wide_res50"]
RES = [0, 1, 2, 3]
ALL_ARCHS = [f"{n}-res{r}" for n in NETS for r in RES]  # 12

# Iperparametri di training (stadio B), griglia regolare; i default della repo stanno sulla griglia
SPACE = {
    "lr_exp": IntDistribution(-7, 1),  # lr = 5e-3 * 2**e  -> 3.9e-5 ... 1e-2
    "L2": IntDistribution(0, 2),       # 0 coseno, 1 MSE, 2 entrambe
    "rate_i": IntDistribution(0, 5),   # 0 = loss ausiliaria spenta (layerloss=0); i>=1: 0.05*2**(i-1)
}                                      #   -> 0, 0.05, 0.1, 0.2, 0.4, 0.8
DEFAULT_IDX = {"lr_exp": 0, "L2": 0, "rate_i": 1}


def decode(p: dict) -> dict:
    rate = 0.0 if p["rate_i"] == 0 else round(0.05 * 2.0 ** (p["rate_i"] - 1), 4)
    return {
        "learning_rate": 5e-3 * 2.0 ** p["lr_exp"],
        "L2": p["L2"],
        "layerloss": 0 if rate == 0.0 else 1,
        "rate": rate,
    }


def check_args(args, parser) -> None:
    pass  # nessun vincolo aggiuntivo (nessuno scheduler LR in SK-RD4AD)


# Hook su main.evaluation: train() la usa come funzione globale del modulo main (importata da test.py).
# Con seg=1 ritorna (auroc_px, auroc_sp, aupro, ap_loc, f1_sp, prec_sp, rec_sp, f1_px).
assert hasattr(skrd_main, "evaluation"), "main.py non espone `evaluation`: hook non applicabile"
train = skrd_main.train
_orig_eval = skrd_main.evaluation
_active_hook = [None]


def _hooked_eval(*a, **kw):
    out = _orig_eval(*a, **kw)
    if _active_hook[0] is not None:
        _active_hook[0](a, kw, out)
    return out


skrd_main.evaluation = _hooked_eval


def count_params_m(encoder, bn, decoder) -> float:
    """Parametri del modello usato in inferenza, in milioni: encoder (senza layer4/fc, che il forward di
    RD4AD non usa), bottleneck e decoder."""
    enc = sum(p.numel() for n, p in encoder.named_parameters() if not n.startswith(("layer4", "fc")))
    return (enc + sum(p.numel() for p in bn.parameters()) + sum(p.numel() for p in decoder.parameters())) / 1e6


def train_once(args, arch, cfg, tag, trial_number, on_eval, set_params) -> dict:
    """Addestra una configurazione. Ritorna {"auroc": AUROC sample del checkpoint scelto, "extras": {...}}."""
    net, res = arch.split("-res")
    ckpt_dir = Path(args.results_dir) / "tmp_ckpt" / f"{tag}_{arch}_t{trial_number}"  # cancellata a fine trial
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.results_dir) / "tmp_imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(args.seed)

    state = {"n": 0, "params": False}

    def hook(a, kw, out):
        state["n"] += 1  # train() valuta a (epoch+1) % print_epoch == 0
        if not state["params"]:  # evaluation(encoder, bn, decoder, ...) riceve i tre moduli
            enc, bn, dec = (a[0], a[1], a[2]) if len(a) >= 3 else (kw["encoder"], kw["bn"], kw["decoder"])
            set_params(count_params_m(enc, bn, dec))
            state["params"] = True
        on_eval(state["n"] * args.eval_every, float(out[1]))

    _active_hook[0] = hook
    try:
        metrics = train(
            class_=args.category, epochs=args.epochs, learning_rate=cfg["learning_rate"], res=int(res),
            batch_size=cfg["batch"], print_epoch=args.eval_every, seg=1, data_path=args.datasets_folder,
            print_canshu=0, score_num=1, print_loss=0, img_path=str(img_dir), vis=0, cut=0,
            layerloss=cfg["layerloss"], rate=cfg["rate"], print_max=0, net=net, L2=cfg["L2"],
            seed=args.seed, aug_cfg=AugConfig(),  # augmentation OFF (enabled=False)
            ckpt_path=str(ckpt_dir), project_name=f"{tag}_t{trial_number}",
        )
    finally:
        _active_hook[0] = None
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    if metrics is None:
        raise optuna.TrialPruned("best_metrics e' rimasto None (nessuna valutazione ha migliorato il punteggio)")

    names = ("auroc_px", "auroc_sp", "aupro", "ap_loc", "f1_sp", "prec_sp", "rec_sp", "f1_px")
    return {"auroc": float(metrics[1]), "extras": {k: float(v) for k, v in zip(names, metrics)}}


# ======================================================================================
# COMMON START -- blocco IDENTICO in hpo_ssn.py e hpo_skrd4ad.py (verificato con diff)
# ======================================================================================
import argparse
import gc
import json
import multiprocessing
import time
from pathlib import Path

import optuna
import torch
from optuna.distributions import IntDistribution
from optuna.trial import TrialState, create_trial

# Il batch size e' trattato allo stesso modo in entrambi gli script: FISSO (--batch) oppure, con
# --tune_batch, ottimizzato sulla griglia 4 8 16 32.
BATCH_SPACE = {"batch_exp": IntDistribution(2, 5)}  # batch = 2**e


def space_of(args) -> dict:
    return {**SPACE, **(BATCH_SPACE if args.tune_batch else {})}


def default_idx_of(args) -> dict:
    idx = dict(DEFAULT_IDX)
    if args.tune_batch:
        idx["batch_exp"] = args.batch.bit_length() - 1
    return idx


def decode_all(p: dict, args) -> dict:
    cfg = decode(p)
    cfg["batch"] = 2 ** p["batch_exp"] if args.tune_batch else args.batch
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Optuna HPO a due stadi ({NAME}): AUROC max, parametri min")
    p.add_argument("--stage", choices=["screen", "tune", "all", "report"], required=True)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--datasets_folder", type=str, required=True)
    p.add_argument("--storage", type=str, default=f"sqlite:///{NAME}_tuning.db",
                   help="Su Colab usa un path locale (/content/...), non Drive")
    p.add_argument("--results_dir", type=str, default="./results_optuna")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Epoche di OGNI trial = epoche della run finale")
    p.add_argument("--eval_every", type=int, default=20, help="Ogni quante epoche si valuta (e si fa pruning)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size fisso (default della repo)")
    p.add_argument("--tune_batch", action="store_true", help="Ottimizza il batch (4 8 16 32) invece di fissarlo")
    p.add_argument("--seed", type=int, default=42, help="Seed fisso dell'HPO (la run finale e' multi-seed)")
    p.add_argument("--vram_limit_gb", type=float, default=15.0,
                   help="Tetto VRAM (GB) per simulare la RTX A4000 da 16 GB; 0 = nessun tetto")
    # selezione e budget
    p.add_argument("--screen_archs", type=str, nargs="+", default=None, choices=ALL_ARCHS,
                   help="Stadio A: sottoinsieme di architetture (default: tutte e 12)")
    p.add_argument("--archs", type=str, nargs="+", default=None, choices=ALL_ARCHS,
                   help="Stadio B: architetture esplicite al posto di quelle scelte dal fronte")
    p.add_argument("--top_k", type=int, default=3, help="Stadio B: architetture portate avanti (max)")
    p.add_argument("--front_delta", type=float, default=0.01,
                   help="Stadio B: si considerano le architetture del fronte con AUROC >= migliore - delta")
    p.add_argument("--n_trials", type=int, default=30, help="Stadio B: trial NUOVI per architettura")
    # pruning
    p.add_argument("--warmup_frac", type=float, default=1 / 3, help="MedianPruner: frazione di epoche senza pruning")
    p.add_argument("--pruner_startup", type=int, default=5, help="MedianPruner: trial completati prima di potare")
    p.add_argument("--pruner_min_trials", type=int, default=3, help="MedianPruner: n_min_trials per step")
    p.add_argument("--sampler_startup", type=int, default=8, help="TPE: trial casuali iniziali")
    p.add_argument("--floor_auroc", type=float, default=0.55,
                   help="Pruning assoluto (solo stadio B): AUROC massimo < soglia dopo --floor_epoch")
    p.add_argument("--floor_epoch", type=int, default=20)
    p.add_argument("--auroc_tolerance", type=float, default=None,
                   help="Report: architettura piu' piccola con AUROC >= migliore - tolleranza")
    args = p.parse_args()

    if args.epochs % args.eval_every != 0:
        p.error("--epochs deve essere multiplo di --eval_every (altrimenti l'ultima epoca non e' valutata)")
    if args.tune_batch and args.batch not in (4, 8, 16, 32):
        p.error("con --tune_batch, --batch (il default dello screening) deve essere 4, 8, 16 o 32")
    check_args(args, p)
    return args


# --------------------------------------------------------------------------------------
# Un trial = un addestramento completo (con eventuale pruning)
# --------------------------------------------------------------------------------------
def run_trial(trial: optuna.Trial, args, arch: str, cfg: dict, tag: str, pruning: bool) -> float:
    """Ritorna l'AUROC di immagine del checkpoint che la repo seleziona. Se `pruning`, a ogni
    valutazione (step = epoca) riporta l'AUROC massimo visto finora e consulta il MedianPruner."""
    trial.set_user_attr("arch", arch)
    trial.set_user_attr("cfg", cfg)
    if torch.cuda.is_available():
        trial.set_user_attr("gpu", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    state = {"n": 0, "best": float("-inf"), "curve": {}}

    def on_eval(step: int, auroc: float) -> None:
        state["n"] += 1
        if step > args.epochs:
            return
        if auroc != auroc:
            raise optuna.TrialPruned("NaN AUROC")
        state["best"] = max(state["best"], auroc)
        state["curve"][str(step)] = state["best"]
        trial.report(state["best"], step)
        if pruning:
            if step >= args.floor_epoch and state["best"] < args.floor_auroc:
                raise optuna.TrialPruned(f"AUROC max {state['best']:.3f} < {args.floor_auroc} a epoca {step}")
            if trial.should_prune():
                raise optuna.TrialPruned(f"MedianPruner a epoca {step} (AUROC max {state['best']:.4f})")

    def set_params(params_m: float) -> None:
        trial.set_user_attr("params_m", round(params_m, 4))

    out = None
    pruned_msg = None
    t0 = time.time()
    try:
        out = train_once(args, arch, cfg, tag, trial.number, on_eval, set_params)
        if state["n"] == 0:
            raise RuntimeError("L'hook di valutazione non e' mai stato chiamato: pruning/curve non attivi")
    except optuna.TrialPruned as e:
        pruned_msg = str(e) or "pruned"
    except RuntimeError as e:
        if "out of memory" in str(e):
            pruned_msg = "CUDA out of memory"
        else:
            raise
    finally:
        # la traceback di un'eccezione in volo terrebbe vivi modello e dataloader: si pulisce qui e il
        # TrialPruned viene rilanciato DOPO, fuori dal blocco except
        trial.set_user_attr("duration_s", round(time.time() - t0, 1))
        if state["curve"]:
            trial.set_user_attr("curve", state["curve"])
        if torch.cuda.is_available():
            trial.set_user_attr("peak_vram_gb", round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        alive = multiprocessing.active_children()
        if alive:
            print(f"ATTENZIONE: {len(alive)} processi figli ancora vivi dopo il trial (worker dei DataLoader?)")

    if pruned_msg is not None:
        print(f"[{tag} | {arch} | trial {trial.number}] PRUNED: {pruned_msg}")
        raise optuna.TrialPruned(pruned_msg)

    auroc = float(out["auroc"])
    if auroc != auroc:
        raise optuna.TrialPruned("NaN AUROC")
    for k, v in out["extras"].items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("auroc_peak", state["best"])
    print(f"\n[{tag} | {arch} | trial {trial.number}] AUROC (ckpt scelto): {auroc:.4f} | picco: {state['best']:.4f} | "
          f"params: {trial.user_attrs.get('params_m', float('nan')):.2f} M | "
          f"{trial.user_attrs['duration_s'] / 60:.1f} min\n")
    return auroc


# --------------------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------------------
def setup_gpu(args) -> None:
    # Il tetto fa fallire con OOM (-> trial potato) cio' che non entrerebbe nella RTX A4000 da 16 GB.
    # Il contesto CUDA (~0.3-0.5 GB) non e' contato dal tetto, da qui il default 15 GB.
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if 0 < args.vram_limit_gb < total_gb:
            torch.cuda.set_per_process_memory_fraction(args.vram_limit_gb / total_gb, 0)
        cap = min(total_gb, args.vram_limit_gb) if args.vram_limit_gb > 0 else total_gb
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB), tetto VRAM: {cap:.1f} GB")
    else:
        print("ATTENZIONE: CUDA non disponibile")


def recover_stale(study: optuna.Study) -> None:
    """Un run interrotto lascia trial RUNNING nel DB: li chiude come FAIL."""
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.RUNNING:
            study.tell(t.number, state=TrialState.FAIL)
            print(f"[{study.study_name}] trial {t.number} era rimasto RUNNING -> FAIL")


def pareto(items):
    """items: lista di (id, auroc, params_m). Fronte: AUROC massimo, parametri minimo (dominanza stretta)."""
    front = []
    for i, a, s in items:
        dominated = any((a2 >= a and s2 <= s) and (a2 > a or s2 < s) for j, a2, s2 in items if j != i)
        if not dominated:
            front.append((i, a, s))
    return sorted(front, key=lambda x: x[2])


def screen_name(args):
    return f"{NAME}_{args.category}_screen"


def tune_name(args, arch):
    return f"{NAME}_{args.category}_tune_{arch}"


def screen_rows(args):
    """Trial di screening completati: (arch, auroc, params_m, trial)."""
    try:
        study = optuna.load_study(study_name=screen_name(args), storage=args.storage)
    except KeyError:
        return []
    return [(t.params["arch"], t.value, t.user_attrs.get("params_m", float("inf")), t)
            for t in study.trials if t.state == TrialState.COMPLETE]


# --------------------------------------------------------------------------------------
# Stadio A: screening delle 12 architetture con iperparametri di default, senza pruning
# --------------------------------------------------------------------------------------
def run_screen(args) -> None:
    study = optuna.create_study(
        study_name=screen_name(args), storage=args.storage, direction="maximize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.NopPruner(),
    )
    recover_stale(study)
    cfg = decode_all(default_idx_of(args), args)

    wanted = [a for a in ALL_ARCHS if args.screen_archs is None or a in args.screen_archs]  # piccolo -> grande
    covered = {
        t.params.get("arch", t.system_attrs.get("fixed_params", {}).get("arch"))
        for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.WAITING)
    }
    for arch in wanted:
        if arch not in covered:
            study.enqueue_trial({"arch": arch})
    pending = sum(t.state == TrialState.WAITING for t in study.trials)
    if pending == 0:
        print("Screening gia' completo")
        return
    print(f"\n=== STADIO A: {pending} architetture ({len(wanted) - pending} gia' fatte), "
          f"{args.epochs} epoche ciascuna, iperparametri di default {cfg} ===")

    def objective(trial):
        arch = trial.suggest_categorical("arch", ALL_ARCHS)
        return run_trial(trial, args, arch, cfg, tag="screen", pruning=False)

    study.optimize(objective, n_trials=pending, gc_after_trial=True)


def select_front(args):
    """Regola (fissata a priori): dal fronte di Pareto dello screening (AUROC max, parametri min) si tengono
    le architetture con AUROC >= migliore - front_delta e, se sono piu' di top_k, le top_k PIU' PICCOLE."""
    rows = screen_rows(args)
    if not rows:
        raise SystemExit("Nessun risultato di screening: lancia prima --stage screen o passa --archs")
    front = pareto([(a, v, s) for a, v, s, _ in rows])
    best = max(v for _, v, _ in front)
    near = [f for f in front if f[1] >= best - args.front_delta]  # gia' ordinato per parametri
    chosen = near[: args.top_k]
    return sorted((c[0] for c in chosen), key=ALL_ARCHS.index), front, best


def report_screen(args) -> None:
    rows = screen_rows(args)
    if not rows:
        print("\n(nessun risultato di screening)")
        return
    front_ids = {f[0] for f in pareto([(a, v, s) for a, v, s, _ in rows])}
    print("\n--- STADIO A: screening (iperparametri di default), ordinato per parametri ---")
    print(f"{'architettura':<28}{'params M':>9}{'AUROC':>9}{'picco':>8}{'VRAM GB':>9}{'min':>7}  fronte")
    for a, v, s, t in sorted(rows, key=lambda r: r[2]):
        u = t.user_attrs
        print(f"{a:<28}{s:>9.2f}{v:>9.4f}{u.get('auroc_peak', float('nan')):>8.4f}"
              f"{u.get('peak_vram_gb', float('nan')):>9.2f}{u.get('duration_s', 0) / 60:>7.1f}  "
              f"{'*' if a in front_ids else ''}")
    print("Nota: 1 seed -> differenze di AUROC ~0.002-0.005 sono dentro il rumore.")


# --------------------------------------------------------------------------------------
# Stadio B: tuning per architettura con MedianPruner (i parametri sono fissi dentro un'architettura)
# --------------------------------------------------------------------------------------
def make_pruner(args):
    warm = int(round(args.epochs * args.warmup_frac / args.eval_every)) * args.eval_every
    return optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup, n_warmup_steps=warm, n_min_trials=args.pruner_min_trials
    ), warm


def warm_start(study: optuna.Study, args, arch: str) -> None:
    """Il trial di default e' gia' stato addestrato nello screening (stesse epoche, stesso seed): lo si copia
    nello studio con la sua curva invece di rifarlo, cosi' entra nella mediana. Se la sua configurazione non
    coincide con il default attuale (es. --batch diverso), il default viene rieseguito."""
    if study.trials:
        return
    idx = default_idx_of(args)
    src = [t for a, _, _, t in screen_rows(args) if a == arch and t.user_attrs.get("cfg") == decode_all(idx, args)]
    if not src:
        study.enqueue_trial(dict(idx))
        return
    t = src[0]
    curve = {int(k): float(v) for k, v in t.user_attrs.get("curve", {}).items()}
    attrs = {k: v for k, v in t.user_attrs.items() if k != "curve"}
    attrs["from_screening"] = True
    study.add_trial(create_trial(
        state=TrialState.COMPLETE, value=t.value, params=dict(idx), distributions=dict(space_of(args)),
        user_attrs=attrs, intermediate_values=curve,
    ))


def run_tune(args, archs) -> None:
    pruner, warm = make_pruner(args)
    space = space_of(args)
    for arch in archs:
        study = optuna.create_study(
            study_name=tune_name(args, arch), storage=args.storage, direction="maximize",
            load_if_exists=True, pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.sampler_startup),
        )
        recover_stale(study)
        warm_start(study, args, arch)
        done = sum(1 for t in study.trials if t.state.is_finished() and not t.user_attrs.get("from_screening"))
        remaining = args.n_trials - done
        if remaining <= 0:
            print(f"[{arch}] gia' {done} trial nuovi, salto")
            continue
        print(f"\n=== STADIO B | {arch}: {remaining} trial ({done} gia' fatti) | "
              f"MedianPruner: startup {args.pruner_startup}, warmup {warm} epoche | "
              f"batch {'ottimizzato' if args.tune_batch else f'fisso a {args.batch}'} ===")

        def objective(trial, arch=arch):
            p = {k: trial.suggest_int(k, d.low, d.high) for k, d in space.items()}
            return run_trial(trial, args, arch, decode_all(p, args), tag="tune", pruning=True)

        study.optimize(objective, n_trials=remaining, gc_after_trial=True)


def report_tune(args, archs) -> None:
    summary = {"model": NAME, "category": args.category, "epochs": args.epochs, "seed": args.seed,
               "batch": "tuned" if args.tune_batch else args.batch, "archs": {}}
    best_rows = []
    print("\n--- STADIO B: risultati per architettura ---")
    for arch in archs:
        try:
            study = optuna.load_study(study_name=tune_name(args, arch), storage=args.storage)
        except KeyError:
            continue
        comp = [t for t in study.trials if t.state == TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
        if not comp:
            continue
        top = sorted(comp, key=lambda t: -t.value)[:3]
        params_m = top[0].user_attrs.get("params_m", float("inf"))
        print(f"\n{arch} | {params_m:.2f} M parametri | completi {len(comp)}, potati {len(pruned)}")
        for t in top:
            tagd = " (default da screening)" if t.user_attrs.get("from_screening") else ""
            print(f"  trial {t.number:>3} | AUROC {t.value:.4f} | {t.user_attrs['cfg']}{tagd}")
        summary["archs"][arch] = {
            "params_m": params_m, "n_complete": len(comp), "n_pruned": len(pruned),
            "top": [{"trial": t.number, "auroc": t.value, "cfg": t.user_attrs["cfg"]} for t in top],
        }
        best_rows.append((arch, top[0].value, params_m))

    if best_rows:
        print("\n--- Fronte tra le architetture (miglior config di ciascuna) ---")
        for a, v, s in pareto(best_rows):
            print(f"  {a:<28} AUROC {v:.4f} | {s:.2f} M")
        if args.auroc_tolerance is not None:
            best = max(v for _, v, _ in best_rows)
            a, v, s = min((r for r in best_rows if r[1] >= best - args.auroc_tolerance), key=lambda r: r[2])
            print(f"\nPiu' piccola con AUROC >= {best:.4f} - {args.auroc_tolerance}: {a} ({v:.4f}, {s:.2f} M)")
        print("\nNota: il miglior trial su ~30 e' ottimistico (winner's curse, 1 seed): la run finale "
              "multi-seed sulle migliori config dice quanto vale davvero.")
        out = Path(args.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"hpo_summary_{NAME}_{args.category}.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"Riepilogo salvato in {path}")


def main():
    args = parse_args()
    setup_gpu(args)
    print(f"HPO {NAME} - {args.category} - {args.epochs} epoche per trial - augmentation OFF")

    if args.stage in ("screen", "all"):
        run_screen(args)
    if args.stage in ("screen", "all", "report"):
        report_screen(args)

    if args.stage in ("tune", "all", "report"):
        if args.archs:
            archs = sorted(args.archs, key=ALL_ARCHS.index)
        elif args.stage == "report":
            archs = ALL_ARCHS  # report: mostra tutti gli studi di tuning esistenti
        else:
            archs, front, best = select_front(args)
            print("\nFronte di Pareto dello screening: " + ", ".join(f[0] for f in front))
            print(f"Regola: fronte con AUROC >= {best:.4f} - {args.front_delta}, al massimo le {args.top_k} piu' piccole")
            print(f"Architetture portate allo stadio B: {archs}")
        if args.stage != "report":
            run_tune(args, archs)
        report_tune(args, archs)


if __name__ == "__main__":
    main()