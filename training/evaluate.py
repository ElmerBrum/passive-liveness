"""
Avaliação de um checkpoint treinado (MultiFTNet) segundo a ISO/IEC 30107-3.

Métricas reportadas (ver docs/09_plano-de-captura-dataset.md, seção 7):

    BPCER = fração de bona fide (live, label 1) classificados como ataque.
    APCER = fração de ataques (label 0) aceitos como bona fide.
            Reportado por espécie de PAI (coluna pai_type do manifest) e o
            PIOR caso  APCER = max_i APCER_i  (ISO/IEC 30107-3), além do geral.
    ACER  = (APCER + BPCER) / 2  — sempre reportamos o par, não só o ACER.

Seleção de limiar (threshold): calculamos o EER na partição de VALIDAÇÃO e
usamos esse limiar fixo para reportar APCER/BPCER/ACER no TESTE. Também
reportamos BPCER @ APCER = 1% e @ 5% (limiares também escolhidos na validação).
Por fim, ROC AUC e a curva DET/ROC (opcionalmente salva em CSV).

Contrato do checkpoint (salvo por train.py):
    torch.save({"model_state":   <MultiFTNet.state_dict()>,
                "backbone_state": <MiniFASNet state_dict>,
                "config":        <asdict(TrainConfig)>}, path)

Em modo eval, MultiFTNet.forward(x) devolve cls_logits[B, 2]; a probabilidade
de LIVE é softmax(logits, dim=1)[:, 1] (label 1 = live/real).

As métricas (EER/APCER/BPCER/AUC) são implementadas em NumPy para evitar uma
dependência dura de scikit-learn; se sklearn estiver disponível ele é usado
apenas como conferência opcional do AUC.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Reusa o contrato compartilhado (paths, tamanhos, hiperparâmetros).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TrainConfig  # noqa: E402
from dataset import SpoofFTDataset  # noqa: E402
from models.multiftnet import build_multiftnet  # noqa: E402

from torch.utils.data import DataLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Métricas ISO/IEC 30107-3 — implementação NumPy (sem dependência de sklearn)
# ---------------------------------------------------------------------------
#
# Convenção de decisão: "live se score >= threshold, senão ataque".
#   score  = softmax(logits)[:, 1]  (probabilidade de bona fide/live)
#   label  = 1 (live/bona fide) | 0 (ataque/spoof)
#
# BPCER(t) = P(score <  t | label == 1)   (bona fide classificado como ataque)
# APCER(t) = P(score >= t | label == 0)   (ataque aceito como bona fide)


def bpcer_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Fração de bona fide (label 1) rejeitados como ataque, no limiar dado."""
    live = scores[labels == 1]
    if live.size == 0:
        return float("nan")
    return float(np.mean(live < threshold))


def apcer_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Fração de ataques (label 0) aceitos como bona fide, no limiar dado."""
    attack = scores[labels == 0]
    if attack.size == 0:
        return float("nan")
    return float(np.mean(attack >= threshold))


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """
    Conjunto de limiares candidatos que cobre todas as transições possíveis
    da curva. Usamos os pontos médios entre scores únicos ordenados, mais os
    extremos (para APCER=0 e BPCER=0). Independe de sklearn.
    """
    uniq = np.unique(scores)
    if uniq.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    lo = uniq[0] - 1e-6
    hi = uniq[-1] + 1e-6
    return np.concatenate(([lo], mids, [hi])).astype(np.float64)


def roc_curve_np(scores: np.ndarray, labels: np.ndarray):
    """
    Curva ROC/DET em termos de PAD: para cada limiar candidato devolve
    (thresholds, apcer, bpcer). apcer = FAR (falso aceite de ataque);
    bpcer = FRR (falsa rejeição de bona fide).
    """
    thresholds = _candidate_thresholds(scores)
    apcer = np.array([apcer_at(scores, labels, t) for t in thresholds])
    bpcer = np.array([bpcer_at(scores, labels, t) for t in thresholds])
    return thresholds, apcer, bpcer


def eer_threshold_np(scores: np.ndarray, labels: np.ndarray):
    """
    Equal Error Rate: limiar onde APCER(t) ≈ BPCER(t). Varremos os limiares
    candidatos e escolhemos o ponto que minimiza |APCER - BPCER| (desempate
    pelo menor máximo dos dois). Devolve (threshold, eer).

    O EER reportado é a média de APCER e BPCER nesse ponto — a definição usual
    quando as duas curvas não se cruzam exatamente por serem discretas.
    """
    thresholds, apcer, bpcer = roc_curve_np(scores, labels)
    diff = np.abs(apcer - bpcer)
    # Desempate: entre limiares com |APCER-BPCER| mínimo, pega o de menor máximo.
    min_diff = diff.min()
    tied = np.where(diff <= min_diff + 1e-12)[0]
    best = tied[np.argmin(np.maximum(apcer[tied], bpcer[tied]))]
    thr = float(thresholds[best])
    eer = float((apcer[best] + bpcer[best]) / 2.0)
    return thr, eer


def threshold_at_apcer(scores: np.ndarray, labels: np.ndarray,
                       target_apcer: float):
    """
    Menor limiar cujo APCER <= target_apcer (operação mais permissiva que ainda
    respeita o teto de APCER). Subir o limiar reduz o APCER, então procuramos o
    menor limiar que já satisfaz o alvo. Devolve (threshold, apcer_atingido).
    """
    thresholds, apcer, _ = roc_curve_np(scores, labels)
    order = np.argsort(thresholds)
    thresholds, apcer = thresholds[order], apcer[order]
    ok = np.where(apcer <= target_apcer + 1e-12)[0]
    if ok.size == 0:
        # Nem no limiar máximo o APCER cai ao alvo; usa o limiar mais rígido.
        idx = int(np.argmin(apcer))
        return float(thresholds[idx]), float(apcer[idx])
    idx = int(ok[0])
    return float(thresholds[idx]), float(apcer[idx])


def auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC AUC via estatística de Mann-Whitney U (probabilidade de um bona fide
    aleatório receber score maior que um ataque aleatório), tratando empates
    com 0.5. Não depende de sklearn.
    """
    pos = scores[labels == 1]  # bona fide
    neg = scores[labels == 0]  # ataque
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank dos scores combinados (empates -> rank médio)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    combined = np.concatenate([pos, neg])[order]
    ranks = _average_ranks(combined)
    # ranks dos positivos: os primeiros n_pos elementos, na ordem original
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    rank_pos = ranks[inv[:n_pos]]
    auc = (rank_pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """Ranks 1..N com rank médio para empates, sobre um vetor já ordenado."""
    n = sorted_vals.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks base-1
        ranks[i:j + 1] = avg
        i = j + 1
    return ranks


def per_pai_apcer(scores: np.ndarray, labels: np.ndarray,
                  pai_types: np.ndarray, threshold: float) -> dict[str, float]:
    """
    APCER por espécie de PAI no limiar dado. Considera apenas amostras de
    ataque (label 0); agrupa pelo pai_type do manifest.
    """
    result: dict[str, float] = {}
    attack_mask = labels == 0
    for pai in sorted(set(pai_types[attack_mask].tolist())):
        mask = attack_mask & (pai_types == pai)
        if not np.any(mask):
            continue
        result[pai] = float(np.mean(scores[mask] >= threshold))
    return result


# ---------------------------------------------------------------------------
# Inferência sobre um split
# ---------------------------------------------------------------------------

def _resolve_manifest_key(path: Path, data_root: Path) -> set[str]:
    """
    Chaves candidatas para casar um sample com uma linha do manifest.
    O manifest pode registrar o filepath de várias formas (absoluto, relativo
    ao data_root, ou só o nome do arquivo), então geramos várias e casamos por
    qualquer coincidência.
    """
    keys = {str(path), path.name}
    try:
        keys.add(str(path.resolve()))
    except OSError:
        pass
    try:
        keys.add(str(path.relative_to(data_root)))
    except ValueError:
        pass
    try:
        keys.add(str(path.resolve().relative_to(data_root.resolve())))
    except (ValueError, OSError):
        pass
    return keys


def load_manifest(manifest_path: Path) -> dict[str, str] | None:
    """
    Lê o manifest.csv e devolve um mapa {chave -> pai_type}, indexado por
    filepath absoluto, relativo e nome-base para maximizar o casamento.
    Retorna None se o arquivo não existe.
    """
    if not manifest_path.is_file():
        return None
    mapping: dict[str, str] = {}
    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "filepath" not in reader.fieldnames:
            warnings.warn(f"manifest sem coluna 'filepath': {manifest_path}")
            return None
        has_pai = "pai_type" in reader.fieldnames
        for row in reader:
            fp = row["filepath"]
            pai = row["pai_type"] if has_pai else "attack"
            p = Path(fp)
            for key in (fp, p.name):
                mapping.setdefault(key, pai)
            try:
                mapping.setdefault(str(p.resolve()), pai)
            except OSError:
                pass
    return mapping


def infer_split(model: torch.nn.Module, cfg: TrainConfig, split: str,
                device: str, batch_size: int = 64):
    """
    Roda o modelo sobre um split e devolve (scores, labels, paths).
      scores : prob. de LIVE = softmax(logits)[:, 1]
      labels : 0 = ataque, 1 = live
      paths  : caminho de cada sample (para casar com o manifest)
    Devolve None se o split não existir no disco.
    """
    try:
        ds = SpoofFTDataset(
            data_root=cfg.data_root,
            patch_info=cfg.patch_info,
            split=split,
            ft_size=cfg.ft_size,
            augment=False,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        warnings.warn(f"Split '{split}' indisponível: {exc}")
        return None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # avaliação: simples e determinístico
        pin_memory=(device != "cpu"),
        drop_last=False,
    )

    scores_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for img_t, _ft_t, label in loader:
            img_t = img_t.to(device)
            logits = model(img_t)                       # eval -> cls_logits[B,2]
            prob_live = torch.softmax(logits, dim=1)[:, 1]
            scores_all.append(prob_live.detach().cpu().numpy())
            labels_all.append(np.asarray(label))

    scores = np.concatenate(scores_all).astype(np.float64)
    labels = np.concatenate(labels_all).astype(np.int64)
    paths = [p for p, _ in ds.samples]
    return scores, labels, paths


# ---------------------------------------------------------------------------
# Carga do checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path, device: str,
                    override_data_root: str | None = None,
                    override_patch_info: str | None = None):
    """
    Carrega o checkpoint no formato do contrato, reconstrói a TrainConfig e o
    MultiFTNet, aplica model_state e coloca em eval(). Devolve (model, cfg).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in ckpt or "config" not in ckpt:
        raise ValueError(
            "Checkpoint fora do contrato: esperado dict com 'model_state' e "
            f"'config' (chaves encontradas: {list(ckpt)})"
        )

    # Reconstrói a TrainConfig só com os campos que ela conhece (robusto a
    # versões antigas/novas do dataclass).
    raw_cfg = dict(ckpt["config"])
    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    # Campos derivados são recalculados no __post_init__; não os passamos.
    derived = {"kernel_size", "ft_size"}
    init_fields = {f.name for f in dataclasses.fields(TrainConfig) if f.init}
    cfg_kwargs = {k: v for k, v in raw_cfg.items()
                  if k in valid and k in init_fields and k not in derived}
    # tuples viram listas ao serem serializados em JSON — reconverte os campos
    # que a TrainConfig espera como tupla.
    for tuple_field in ("input_size", "milestones"):
        if tuple_field in cfg_kwargs and isinstance(cfg_kwargs[tuple_field], list):
            cfg_kwargs[tuple_field] = tuple(cfg_kwargs[tuple_field])
    cfg = TrainConfig(**cfg_kwargs)

    if override_data_root is not None:
        cfg.data_root = override_data_root
    if override_patch_info is not None:
        cfg.patch_info = override_patch_info

    model = build_multiftnet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def format_report(summary: dict) -> str:
    """Monta a tabela textual legível a partir do dict de resultados."""
    lines: list[str] = []
    w = 60
    lines.append("=" * w)
    lines.append("AVALIAÇÃO PAD — ISO/IEC 30107-3")
    lines.append("=" * w)
    lines.append(f"checkpoint : {summary['checkpoint']}")
    lines.append(f"data_root  : {summary['data_root']}")
    lines.append(f"patch_info : {summary['patch_info']}")
    lines.append(f"device     : {summary['device']}")
    lines.append("-" * w)

    val = summary.get("val")
    if val is None:
        lines.append("VAL split ausente — não foi possível calibrar o limiar.")
        lines.append("(APCER/BPCER de teste reportados no limiar 0.5.)")
    else:
        lines.append(f"VAL: n={val['n']}  live={val['n_live']}  attack={val['n_attack']}")
        lines.append(f"  EER threshold     : {val['eer_threshold']:.6f}")
        lines.append(f"  EER               : {val['eer']:.4%}")
        lines.append(f"  AUC (val)         : {val['auc']:.4f}")
        lines.append(f"  thr @ APCER=1%    : {val['thr_apcer1']:.6f} "
                     f"(APCER val {val['apcer1_val']:.4%})")
        lines.append(f"  thr @ APCER=5%    : {val['thr_apcer5']:.6f} "
                     f"(APCER val {val['apcer5_val']:.4%})")
    lines.append("-" * w)

    test = summary.get("test")
    if test is None:
        lines.append("TEST split ausente — nada a reportar em teste.")
        lines.append("=" * w)
        return "\n".join(lines)

    lines.append(f"TEST: n={test['n']}  live={test['n_live']}  attack={test['n_attack']}")
    lines.append(f"  threshold usado   : {test['threshold']:.6f} "
                 f"({test['threshold_source']})")
    lines.append(f"  AUC (test)        : {test['auc']:.4f}")
    lines.append("")
    lines.append("  @ EER threshold (calibrado na val):")
    lines.append(f"    APCER (overall) : {test['apcer_overall']:.4%}")
    lines.append(f"    APCER (worst)   : {test['apcer_worst']:.4%} "
                 f"[{test['apcer_worst_pai']}]")
    lines.append(f"    BPCER           : {test['bpcer']:.4%}")
    lines.append(f"    ACER            : {test['acer']:.4%}")
    if test.get("per_pai_apcer"):
        lines.append("    APCER por PAI:")
        for pai, val_ in sorted(test["per_pai_apcer"].items()):
            lines.append(f"      - {pai:<18}: {val_:.4%}")
    else:
        lines.append("    (manifest indisponível — só APCER overall)")
    lines.append("")
    lines.append("  Operating points (limiar calibrado na val):")
    if test.get("bpcer_at_apcer1") is not None:
        lines.append(f"    BPCER @ APCER=1%  : {test['bpcer_at_apcer1']:.4%}")
        lines.append(f"    BPCER @ APCER=5%  : {test['bpcer_at_apcer5']:.4%}")
    else:
        lines.append("    (val ausente — operating points não calculados)")
    lines.append("=" * w)
    return "\n".join(lines)


def evaluate(checkpoint: str, data_root: str | None, patch_info: str | None,
             device: str, manifest: str | None, batch_size: int = 64) -> dict:
    """
    Pipeline completo de avaliação. Devolve um dict serializável com todas as
    métricas (também usado pela saída --out e pela impressão).
    """
    ckpt_path = Path(checkpoint)
    model, cfg = load_checkpoint(
        ckpt_path, device,
        override_data_root=data_root,
        override_patch_info=patch_info,
    )

    data_root_p = Path(cfg.data_root)
    manifest_path = Path(manifest) if manifest else data_root_p / "manifest.csv"
    manifest_map = load_manifest(manifest_path)
    if manifest_map is None:
        warnings.warn(
            f"manifest não encontrado/inválido em {manifest_path}; "
            "APCER por PAI será omitido (só overall)."
        )

    summary: dict = {
        "checkpoint": str(ckpt_path),
        "data_root": str(cfg.data_root),
        "patch_info": cfg.patch_info,
        "device": device,
        "manifest": str(manifest_path) if manifest_map is not None else None,
        "val": None,
        "test": None,
    }

    # --- VAL: calibra o limiar ---
    val = infer_split(model, cfg, "val", device, batch_size)
    eer_thr = 0.5
    thr_apcer1 = thr_apcer5 = None
    if val is not None:
        v_scores, v_labels, _ = val
        eer_thr, eer = eer_threshold_np(v_scores, v_labels)
        thr_apcer1, apcer1_val = threshold_at_apcer(v_scores, v_labels, 0.01)
        thr_apcer5, apcer5_val = threshold_at_apcer(v_scores, v_labels, 0.05)
        summary["val"] = {
            "n": int(v_scores.size),
            "n_live": int(np.sum(v_labels == 1)),
            "n_attack": int(np.sum(v_labels == 0)),
            "eer_threshold": eer_thr,
            "eer": eer,
            "auc": auc_np(v_scores, v_labels),
            "thr_apcer1": thr_apcer1,
            "apcer1_val": apcer1_val,
            "thr_apcer5": thr_apcer5,
            "apcer5_val": apcer5_val,
        }

    # --- TEST: reporta no limiar fixo calibrado na val ---
    test = infer_split(model, cfg, "test", device, batch_size)
    if test is not None:
        t_scores, t_labels, t_paths = test

        # pai_type por sample (via manifest); "attack"/"live" como fallback.
        if manifest_map is not None:
            pai_types = np.array([
                _lookup_pai(manifest_map, p, data_root_p, lbl)
                for p, lbl in zip(t_paths, t_labels)
            ])
        else:
            pai_types = np.array(["live" if lbl == 1 else "attack"
                                  for lbl in t_labels])

        apcer_overall = apcer_at(t_scores, t_labels, eer_thr)
        bpcer = bpcer_at(t_scores, t_labels, eer_thr)
        acer = float((apcer_overall + bpcer) / 2.0)

        pp = per_pai_apcer(t_scores, t_labels, pai_types, eer_thr) \
            if manifest_map is not None else {}
        if pp:
            worst_pai = max(pp, key=pp.get)
            apcer_worst = pp[worst_pai]
        else:
            worst_pai = "overall"
            apcer_worst = apcer_overall

        threshold_source = "EER@val" if val is not None else "0.5 (val ausente)"

        bpcer_apcer1 = bpcer_apcer5 = None
        if thr_apcer1 is not None:
            bpcer_apcer1 = bpcer_at(t_scores, t_labels, thr_apcer1)
            bpcer_apcer5 = bpcer_at(t_scores, t_labels, thr_apcer5)

        summary["test"] = {
            "n": int(t_scores.size),
            "n_live": int(np.sum(t_labels == 1)),
            "n_attack": int(np.sum(t_labels == 0)),
            "threshold": eer_thr,
            "threshold_source": threshold_source,
            "auc": auc_np(t_scores, t_labels),
            "apcer_overall": float(apcer_overall),
            "bpcer": float(bpcer),
            "acer": acer,
            "apcer_worst": float(apcer_worst),
            "apcer_worst_pai": worst_pai,
            "per_pai_apcer": pp,
            "bpcer_at_apcer1": bpcer_apcer1,
            "bpcer_at_apcer5": bpcer_apcer5,
        }
        # curva ROC/DET para CSV (sobre o teste)
        thr, apcer_c, bpcer_c = roc_curve_np(t_scores, t_labels)
        summary["_roc"] = {
            "threshold": thr.tolist(),
            "apcer": apcer_c.tolist(),
            "bpcer": bpcer_c.tolist(),
        }

    return summary


def _lookup_pai(manifest_map: dict[str, str], path: Path,
                data_root: Path, label: int) -> str:
    """Casa um sample com o pai_type do manifest por qualquer chave candidata."""
    for key in _resolve_manifest_key(path, data_root):
        if key in manifest_map:
            return manifest_map[key]
    # Sem match: usa a semântica do label (não quebra o APCER overall).
    return "live" if label == 1 else "unknown-attack"


def write_outputs(summary: dict, out_path: Path) -> None:
    """
    Grava:
      <out>.json  — summary completo (sem a curva ROC, que vai no CSV)
      <out>.txt   — a tabela legível
      <out>_roc.csv — threshold,apcer,bpcer (se houver teste)
    Se `out` tiver sufixo, ele é respeitado como base.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.with_suffix("")

    roc = summary.pop("_roc", None)

    json_path = base.with_suffix(".json")
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    txt_path = base.with_suffix(".txt")
    with txt_path.open("w") as fh:
        fh.write(format_report(summary) + "\n")

    if roc is not None:
        csv_path = Path(str(base) + "_roc.csv")
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["threshold", "apcer", "bpcer"])
            for t, a, b in zip(roc["threshold"], roc["apcer"], roc["bpcer"]):
                writer.writerow([f"{t:.8f}", f"{a:.8f}", f"{b:.8f}"])
        print(f"[out] ROC/DET -> {csv_path}")
    print(f"[out] summary  -> {json_path}")
    print(f"[out] report   -> {txt_path}")


# ---------------------------------------------------------------------------
# Self-test (sandbox sem GPU e sem checkpoint treinado)
# ---------------------------------------------------------------------------

def _run_self_test() -> int:
    """
    Verifica o pipeline ponta-a-ponta em CPU:
      1. constrói um MultiFTNet fresco (não treinado) e salva no formato-contrato;
      2. fabrica um mini dataset preparado (pngs 80x80 BGR em val/ e test/, com
         classes 0/1) + um manifest.csv coerente;
      3. roda evaluate() e verifica que todas as métricas são finitas.
    Não valida qualidade (modelo é aleatório) — só a integridade do fluxo.
    """
    import tempfile

    import cv2  # importado aqui: só necessário no self-test

    torch.manual_seed(0)
    np.random.seed(0)

    tmp = Path(tempfile.mkdtemp(prefix="pl_eval_selftest_"))
    print(f"[self-test] tmp dir: {tmp}")

    cfg = TrainConfig(
        data_root=str(tmp / "data"),
        patch_info="2.7_80x80",
        device="cpu",
        num_workers=0,
        batch_size=8,
    )

    # (1) checkpoint no formato do contrato
    model = build_multiftnet(cfg)
    ckpt_path = tmp / "best.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone_state": model.backbone_state_dict(),
            "config": dataclasses.asdict(cfg),
        },
        ckpt_path,
    )
    print(f"[self-test] checkpoint salvo: {ckpt_path}")

    # (2) mini dataset preparado + manifest
    data_root = Path(cfg.data_root)
    manifest_rows: list[dict] = []
    pai_for_class0 = ["print-matte", "replay-phone"]  # 2 espécies de PAI

    def _make_png(path: Path, seed: int):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)  # BGR
        cv2.imwrite(str(path), img)

    n_per = 6
    for split in ("val", "test"):
        for cls in (0, 1):
            cls_dir = data_root / cfg.patch_info / split / str(cls)
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per):
                fname = f"{split}_{cls}_{i:03d}.png"
                fpath = cls_dir / fname
                _make_png(fpath, seed=hash((split, cls, i)) % (2**31))
                if cls == 1:
                    pai = "live"
                else:
                    pai = pai_for_class0[i % len(pai_for_class0)]
                manifest_rows.append({
                    "filepath": str(fpath),
                    "subject_id": f"S{i:03d}",
                    "label": cls,
                    "pai_type": pai,
                    "split": split,
                })

    manifest_path = data_root / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filepath", "subject_id", "label",
                            "pai_type", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[self-test] manifest: {manifest_path} ({len(manifest_rows)} linhas)")

    # (3) avaliação + asserts de finitude
    summary = evaluate(
        checkpoint=str(ckpt_path),
        data_root=str(data_root),
        patch_info=cfg.patch_info,
        device="cpu",
        manifest=str(manifest_path),
        batch_size=8,
    )

    print("\n" + format_report({k: v for k, v in summary.items()
                                 if k != "_roc"}))

    # asserts
    assert summary["val"] is not None, "val deveria existir"
    assert summary["test"] is not None, "test deveria existir"

    def _finite(x, name):
        assert x is not None and np.isfinite(x), f"{name} não finito: {x}"

    v = summary["val"]
    _finite(v["eer_threshold"], "val.eer_threshold")
    _finite(v["eer"], "val.eer")
    _finite(v["auc"], "val.auc")
    _finite(v["thr_apcer1"], "val.thr_apcer1")
    _finite(v["thr_apcer5"], "val.thr_apcer5")

    t = summary["test"]
    for key in ("apcer_overall", "bpcer", "acer", "apcer_worst", "auc",
                "bpcer_at_apcer1", "bpcer_at_apcer5"):
        _finite(t[key], f"test.{key}")
    assert 0.0 <= t["apcer_overall"] <= 1.0
    assert 0.0 <= t["bpcer"] <= 1.0
    assert t["per_pai_apcer"], "APCER por PAI deveria estar populado"
    for pai, val_ in t["per_pai_apcer"].items():
        _finite(val_, f"per_pai_apcer[{pai}]")
    # o pior PAI deve ser >= APCER overall (é um máximo por espécie)
    assert t["apcer_worst"] >= t["apcer_overall"] - 1e-9

    # sanidade das funções de métrica em casos-limite
    _sanity_metric_units()

    print("\n[self-test] OK — todas as métricas finitas e no intervalo esperado.")
    return 0


def _sanity_metric_units() -> None:
    """Casos determinísticos para as funções de métrica (independem do modelo)."""
    # separação perfeita: ataques com score baixo, live com score alto.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    thr, eer = eer_threshold_np(scores, labels)
    assert abs(eer) < 1e-9, f"EER deveria ser 0 na separação perfeita: {eer}"
    assert abs(auc_np(scores, labels) - 1.0) < 1e-9, "AUC deveria ser 1.0"
    assert apcer_at(scores, labels, thr) == 0.0
    assert bpcer_at(scores, labels, thr) == 0.0

    # score aleatório inverso -> AUC 0
    assert abs(auc_np(scores, 1 - labels)) < 1e-9

    # empates: metade-metade -> AUC 0.5
    s = np.array([0.5, 0.5, 0.5, 0.5])
    l = np.array([0, 1, 0, 1])
    assert abs(auc_np(s, l) - 0.5) < 1e-9

    # comparação opcional com sklearn, se disponível
    try:
        from sklearn.metrics import roc_auc_score
        ref = roc_auc_score(labels, scores)
        assert abs(ref - auc_np(scores, labels)) < 1e-9, "AUC diverge do sklearn"
    except Exception:
        pass  # sklearn opcional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    default_cfg = TrainConfig()
    p = argparse.ArgumentParser(
        description="Avalia um checkpoint MultiFTNet (ISO/IEC 30107-3: "
                    "APCER/BPCER/ACER, EER na val, BPCER@APCER, ROC/AUC).",
    )
    p.add_argument("--checkpoint", type=str,
                   help="caminho do checkpoint (.pth) no formato-contrato.")
    p.add_argument("--data-root", type=str, default=None,
                   help="raiz dos dados preparados "
                        "(default: o do checkpoint / config).")
    p.add_argument("--patch-info", type=str, default=None,
                   help="pasta de escala/tamanho (default: o do checkpoint).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="dispositivo torch (default: cuda se disponível).")
    p.add_argument("--manifest", type=str, default=None,
                   help="manifest.csv (default: <data-root>/manifest.csv).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="batch da avaliação (default: 64).")
    p.add_argument("--out", type=str, default=None,
                   help="prefixo de saída: grava .json/.txt/_roc.csv.")
    p.add_argument("--self-test", action="store_true",
                   help="roda o self-test em CPU (fabrica checkpoint e dados "
                        "temporários) e sai; ignora os demais argumentos.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        return _run_self_test()

    if not args.checkpoint:
        print("erro: --checkpoint é obrigatório (ou use --self-test).",
              file=sys.stderr)
        return 2

    # cuda pedido mas indisponível -> cai para cpu com aviso.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA indisponível; usando CPU.")
        device = "cpu"

    summary = evaluate(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        patch_info=args.patch_info,
        device=device,
        manifest=args.manifest,
        batch_size=args.batch_size,
    )

    print(format_report({k: v for k, v in summary.items() if k != "_roc"}))

    if args.out:
        write_outputs(summary, Path(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
