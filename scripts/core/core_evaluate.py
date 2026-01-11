# scripts/core/core_evaluate.py

import argparse
import csv

# Très gros corpus TSV : sécuriser la taille maximale de champ CSV.
try:
    csv.field_size_limit(2**31 - 1)
except Exception:
    pass
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

from scripts.core.core_utils import (
    resolve_profile_base,
    debug_print_params,
    PIPELINE_VERSION,
    apply_global_seed,
    log,
    parse_seed,

)


#  CLI


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V4 core_evaluate : évaluation multi-familles (spaCy, sklearn, HF, check)"
    )
    ap.add_argument(
        "--profile",
        required=True,
        help="Nom du profil (sans .yml) dans configs/profiles/",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: view=ideology_global)",
    )
    ap.add_argument(
        "--only-family",
        choices=["spacy", "sklearn", "hf", "check"],
        help="Limiter l'évaluation à une seule famille (optionnel)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


#  Utils généraux


def set_blas_threads(n_threads: int) -> None:
    """
    Limiter les threads BLAS (MKL/OPENBLAS/OMP) pour éviter la sur-souscription.
    Même en éval, ça peut éviter des surprises.
    """
    if n_threads is None or n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"[core_evaluate] BLAS threads fixés à {n_threads}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    dataset_id_models, _ = resolve_dataset_ids(params)
    view = params.get("view", "unknown_view")
    return Path("models") / str(dataset_id_models) / view / family / model_id


def get_reports_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    dataset_id_models, _ = resolve_dataset_ids(params)
    view = params.get("view", "unknown_view")
    return Path("reports") / str(dataset_id_models) / view / family / model_id



def load_job_tsv(params: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Charger job.tsv (ou fallback train.tsv) depuis data/interim/{dataset_id}/{view}/
    Retourne (texts, labels, rows_complets).
    """
    dataset_id_models, dataset_id_eval = resolve_dataset_ids(params)
    view = params.get("view", "unknown_view")
    interim_dir = Path("data") / "interim" / str(dataset_id_eval) / view
    job_path = interim_dir / "job.tsv"
    dev_path = interim_dir / "dev.tsv"
    test_path = interim_dir / "test.tsv"
    train_path = interim_dir / "train.tsv"

    # Priorité: test -> job -> dev -> train
    if test_path.exists():
        target = test_path
    elif job_path.exists():
        target = job_path
    elif dev_path.exists():
        target = dev_path
    elif train_path.exists():
        target = train_path
        print("[core_evaluate] WARNING: test/job/dev absents, évaluation réalisée sur train.tsv (sur-apprentissage possible).")
    else:
        raise SystemExit(f"[core_evaluate] Aucune TSV trouvée dans {interim_dir} (train/dev/test/job).")

    texts: List[str] = []
    labels: List[str] = []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("text") or ""
            label = row.get("label")
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
            rows.append(row)

    if not texts:
        raise SystemExit(f"[core_evaluate] Aucune donnée valide dans {target}")

    return texts, labels, rows


def maybe_debug_subsample_eval(
    texts: List[str],
    labels: List[str],
    params: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Si debug_mode=True, limiter la taille du dataset d'évaluation.
    Si rows est fourni, il est sous-échantillonné de manière cohérente.
    """
    if not params.get("debug_mode"):
        return (texts, labels, rows) if rows is not None else (texts, labels)

    max_docs = 1000
    if len(texts) <= max_docs:
        return (texts, labels, rows) if rows is not None else (texts, labels)

    seed = parse_seed(params.get("seed"), default=42) or 42
    print(f"[core_evaluate] debug_mode actif : sous-échantillon de {max_docs} docs sur {len(texts)} (seed={seed})")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    idx_sel = sorted(indices[:max_docs])
    texts_sub = [texts[i] for i in idx_sel]
    labels_sub = [labels[i] for i in idx_sel]
    if rows is not None:
        rows_sub = [rows[i] for i in idx_sel]
        return texts_sub, labels_sub, rows_sub
    return texts_sub, labels_sub


def compute_basic_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Métriques de base (JSON-safe).

    - Fixe l'ordre des labels pour des reports comparables.
    - zero_division=0 pour éviter des erreurs si une classe est absente.
    """
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report_dict,
    }


def group_indices_by_field(rows: List[Dict[str, Any]], field: str) -> Dict[Any, List[int]]:
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        val = row.get(field)
        if val is None:
            val = "__missing__"
        groups[val].append(i)
    return groups


def compute_grouped_metrics(
    rows: Optional[List[Dict[str, Any]]],
    y_true: List[str],
    y_pred: List[str],
    fields: List[str],
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    if not rows or not fields:
        return {}

    if not (len(rows) == len(y_true) == len(y_pred)):
        raise ValueError(
            "[core_evaluate] Longueurs incohérentes pour le calcul des métriques groupées"
        )

    grouped: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for field in fields:
        groups = group_indices_by_field(rows, field)
        metrics_by_field: Dict[Any, Dict[str, Any]] = {}
        for val, idxs in groups.items():
            y_true_g = [y_true[i] for i in idxs]
            y_pred_g = [y_pred[i] for i in idxs]
            metrics_by_field[val] = compute_basic_metrics(y_true_g, y_pred_g)
        grouped[field] = metrics_by_field

    return grouped


def _normalize_weights(raw_weights: dict, group_keys: list) -> dict:
    """Normalise des poids sur un ensemble de groupes.

    - Si raw_weights est vide ou somme à 0, fallback en poids égaux.
    - Les clés manquantes dans raw_weights reçoivent 0 (ignorées), sauf si
      tous les poids sont à 0 auquel cas fallback égal.
    """
    if not group_keys:
        return {}

    if not raw_weights:
        w = {k: 1.0 for k in group_keys}
    else:
        w = {k: float(raw_weights.get(k, 0.0) or 0.0) for k in group_keys}
        if sum(w.values()) <= 0:
            w = {k: 1.0 for k in group_keys}

    s = sum(w.values())
    return {k: (w[k] / s if s > 0 else 0.0) for k in group_keys}


def compute_balanced_summary_by_field(
    grouped_metrics: Dict[str, Dict[Any, Dict[str, Any]]],
    field: str,
    weights_map: Optional[Dict[Any, float]] = None,
) -> Dict[str, Any]:
    """Agrège des métriques par groupes avec pondération.

    Cas d'usage principal : évaluation multi-corpus (WEB+ASR), où les métriques
    globales peuvent être dominées par le corpus majoritaire.

    Stratégie :
      1) on s'appuie sur grouped_metrics[field] (métriques déjà calculées par groupe)
      2) on agrège des scalaires (accuracy, macro_f1) via des poids normalisés
         (par défaut : poids égaux par groupe).
    """
    if not grouped_metrics or field not in grouped_metrics:
        return {}

    by_val = grouped_metrics[field]
    group_keys = list(by_val.keys())
    if not group_keys:
        return {}

    w_norm = _normalize_weights(weights_map or {}, group_keys)

    def _scalar(m: Dict[str, Any], key: str) -> float:
        try:
            return float(m.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    acc = 0.0
    macro_f1 = 0.0
    per_group: Dict[str, Any] = {}

    for g in group_keys:
        m = by_val.get(g, {})
        per_group[str(g)] = {
            "accuracy": _scalar(m, "accuracy"),
            "macro_f1": _scalar(m, "macro_f1"),
            "weight": float(w_norm.get(g, 0.0) or 0.0),
        }
        acc += per_group[str(g)]["weight"] * per_group[str(g)]["accuracy"]
        macro_f1 += per_group[str(g)]["weight"] * per_group[str(g)]["macro_f1"]

    return {
        "field": field,
        "weights_normalized": {str(k): float(v) for k, v in w_norm.items()},
        "balanced_accuracy": acc,
        "balanced_macro_f1": macro_f1,
        "per_group": per_group,
    }


def maybe_attach_balanced_metrics(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    grouped_metrics: Dict[str, Dict[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Ajoute des métriques équilibrées par corpus pour l'analyse multi-corpus.

    Déclencheur : analysis.compare_by contient 'corpus_id'.

    Config optionnelle :
      analysis.group_weights: {corpus_id: {web1: 1.0, asr1: 1.0}}

    Si group_weights est absent : poids égaux par corpus.
    """
    analysis_cfg = params.get("analysis") or {}
    compare_by = analysis_cfg.get("compare_by") or []
    if "corpus_id" not in compare_by:
        return {}

    weights_map = None
    gw = analysis_cfg.get("group_weights")
    if isinstance(gw, dict) and isinstance(gw.get("corpus_id"), dict):
        weights_map = gw.get("corpus_id")

    summary = compute_balanced_summary_by_field(grouped_metrics, "corpus_id", weights_map=weights_map)
    if not summary:
        return {}

    metrics["balanced_accuracy_by_corpus_id"] = summary.get("balanced_accuracy")
    metrics["balanced_macro_f1_by_corpus_id"] = summary.get("balanced_macro_f1")
    metrics["balanced_weights_by_corpus_id"] = summary.get("weights_normalized")

    return summary


def save_balanced_summary(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    summary: Dict[str, Any],
) -> None:
    if not summary:
        return

    reports_dir = get_reports_dir(params, family, model_id)
    ensure_dir(reports_dir)

    field = summary.get("field") or "group"
    path_out = reports_dir / f"metrics_balanced_by_{field}.json"
    with path_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[core_evaluate] metrics_balanced_by_{field}.json écrit : {path_out}")



def save_grouped_metrics(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    grouped_metrics: Dict[str, Dict[Any, Dict[str, Any]]],
) -> None:
    if not grouped_metrics:
        return

    reports_dir = get_reports_dir(params, family, model_id)
    ensure_dir(reports_dir)

    for field, metrics_by in grouped_metrics.items():
        path_group = reports_dir / f"metrics_by_{field}.json"
        with path_group.open("w", encoding="utf-8") as f:
            json.dump(metrics_by, f, ensure_ascii=False, indent=2)
        print(f"[core_evaluate] metrics_by_{field}.json écrit : {path_group}")


def save_eval_outputs(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    metrics: Dict[str, Any],
) -> None:
    reports_dir = get_reports_dir(params, family, model_id)
    ensure_dir(reports_dir)

    # metrics.json : numérique + report dict
    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[core_evaluate] metrics.json écrit : {metrics_path}")

    # classification_report.txt : version textuelle
    # On regénère un report texte à partir du dict pour avoir quelque chose de lisible
    report_txt_path = reports_dir / "classification_report.txt"
    with report_txt_path.open("w", encoding="utf-8") as f:
        # Reconstruction simple du report global
        if "classification_report" in metrics:
            # On pourrait reformatter, mais pour V4-v1 on fait simple:
            f.write(json.dumps(metrics["classification_report"], ensure_ascii=False, indent=2))
        else:
            f.write("No classification_report field in metrics.\n")
    print(f"[core_evaluate] classification_report.txt écrit : {report_txt_path}")

    # meta_eval.json : contexte de l'évaluation
    dataset_id_models, dataset_id_eval = resolve_dataset_ids(params)
    meta_eval = {
        "profile": params.get("profile"),
        "dataset_id": dataset_id_models,          # où les modèles ont été entraînés
        "eval_dataset_id": dataset_id_eval,       # sur quoi on évalue effectivement
        "corpus_id": params.get("corpus_id", params.get("corpus", {}).get("corpus_id")),
        "view": params.get("view"),
        "family": family,
        "model_id": model_id,
        "n_eval_docs": int(metrics.get("n_eval_docs", 0)),
        "pipeline_version": PIPELINE_VERSION,
    }

    meta_eval_path = reports_dir / "meta_eval.json"
    with meta_eval_path.open("w", encoding="utf-8") as f:
        json.dump(meta_eval, f, ensure_ascii=False, indent=2)
    print(f"[core_evaluate] meta_eval.json écrit : {meta_eval_path}")

def resolve_dataset_ids(params: Dict[str, Any]) -> Tuple[str, str]:
    """
    Retourne (dataset_id_for_models, dataset_id_for_eval).

    - dataset_id_for_models : là où sont stockés les modèles (train).
    - dataset_id_for_eval   : dataset utilisé pour job.tsv (eval).
      Par défaut = dataset_id_for_models, sauf si eval_dataset_id est défini.
    """
    corpus_id = params.get("corpus_id", params.get("corpus", {}).get("corpus_id", "unknown_corpus"))
    dataset_id_models = params.get("dataset_id") or corpus_id
    dataset_id_eval = params.get("eval_dataset_id") or dataset_id_models
    return str(dataset_id_models), str(dataset_id_eval)



#  Éval spaCy


def eval_spacy_model(params: Dict[str, Any], model_id: str) -> None:
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError:
        raise SystemExit("[core_evaluate] spaCy n'est pas installé, impossible d'évaluer la famille 'spacy'.")

    model_dir = get_model_output_dir(params, "spacy", model_id)
    if not model_dir.exists():
        print(f"[core_evaluate:spacy] Modèle spaCy introuvable: {model_dir}, skip.")
        return

    load_dir = model_dir
    for cand in ("model-best", "model-last"):
        cand_dir = model_dir / cand
        if cand_dir.exists():
            load_dir = cand_dir
            break

    print(f"[core_evaluate:spacy] Chargement modèle depuis {load_dir}")
    nlp = spacy.load(load_dir)
    # On reconstruit le chemin vers les DocBin éventuels produits par core_prepare
    dataset_id_models, dataset_id_eval = resolve_dataset_ids(params)
    view = params.get("view", "unknown_view")
    spacy_proc_dir = Path("data") / "processed" / str(dataset_id_eval) / view / "spacy"

    # Chercher des DocBin (test/job/dev) (shardés ou non)
    chosen = None
    docbins = []
    for split_name, pat in (("test", "test*.spacy"), ("job", "job*.spacy"), ("dev", "dev*.spacy")):
        L = sorted(spacy_proc_dir.glob(pat))
        if L:
            chosen = split_name
            docbins = L
            break
    if docbins:
        print(
            f"[core_evaluate:spacy] DocBins détectés ({len(docbins)} fichier(s) {chosen}*.spacy) dans {spacy_proc_dir} "
            "→ évaluation via TSV (job/test/dev) afin de conserver les métriques groupées (compare_by)."
        )
    else:
        print("[core_evaluate:spacy] Aucun DocBin test/job/dev*.spacy trouvé, fallback sur TSV")
    texts, labels_true, rows = load_job_tsv(params)
    texts, labels_true, rows = maybe_debug_subsample_eval(texts, labels_true, params, rows)

    # Long texts : même logique que dans core_prepare (DocBin).
    # Objectif : ne jamais planter sur E088/MemoryError, et garder une trace simple de ce qui a été ignoré.
    spacy_max_chars = int(params.get("spacy_max_chars") or 1_000_000)
    spacy_long_text_policy = str(params.get("spacy_long_text_policy") or "truncate").strip().lower()
    if spacy_long_text_policy not in {"keep", "truncate", "drop"}:
        spacy_long_text_policy = "truncate"

    # Empêche spaCy de lever E088 sur les très longs textes.
    nlp.max_length = max(nlp.max_length, spacy_max_chars)

    kept_texts: List[str] = []
    kept_true: List[str] = []
    kept_rows: List[Dict[str, Any]] = []
    labels_pred: List[str] = []
    dropped_long = 0
    truncated_long = 0
    dropped_error = 0

    for txt, y, row in zip(texts, labels_true, rows):
        if txt is None:
            txt = ""
        if len(txt) > spacy_max_chars:
            if spacy_long_text_policy == "drop":
                dropped_long += 1
                continue
            if spacy_long_text_policy == "truncate":
                txt = txt[:spacy_max_chars]
                truncated_long += 1
            # keep : on ne touche pas

        try:
            doc = nlp(txt)
        except (ValueError, MemoryError) as e:
            # E088 (text too long) ou saturation mémoire : on droppe et on continue.
            dropped_error += 1
            continue

        kept_texts.append(txt)
        kept_true.append(y)
        kept_rows.append(row)

        if not doc.cats:
            labels_pred.append("__NO_PRED__")
        else:
            best_label = max(doc.cats.items(), key=lambda kv: kv[1])[0]
            labels_pred.append(best_label)

    print(f"[core_evaluate:spacy] Évaluation sur {len(kept_texts)} docs.")
    if dropped_long or truncated_long or dropped_error:
        print(
            "[core_evaluate:spacy] Long docs: "
            f"policy={spacy_long_text_policy}, max_chars={spacy_max_chars} → "
            f"drop_long={dropped_long}, trunc={truncated_long}, drop_error={dropped_error}."
        )

    metrics = compute_basic_metrics(kept_true, labels_pred)
    metrics["family"] = "spacy"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(kept_texts)
    metrics["dropped_long_docs"] = int(dropped_long)
    metrics["truncated_long_docs"] = int(truncated_long)
    metrics["dropped_on_error"] = int(dropped_error)

    analysis_cfg = params.get("analysis") or {}
    compare_by = analysis_cfg.get("compare_by") or []

    try:
        grouped_metrics = compute_grouped_metrics(kept_rows, kept_true, labels_pred, compare_by)
    except ValueError as e:
        print(str(e))
        grouped_metrics = {}

    balanced_summary = maybe_attach_balanced_metrics(params, metrics, grouped_metrics)

    save_eval_outputs(params, "spacy", model_id, metrics)
    save_grouped_metrics(params, "spacy", model_id, grouped_metrics)
    save_balanced_summary(params, "spacy", model_id, balanced_summary)



#  Éval sklearn


def eval_sklearn_model(params: Dict[str, Any], model_id: str) -> None:
    model_dir = get_model_output_dir(params, "sklearn", model_id)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        print(f"[core_evaluate:sklearn] Modèle sklearn introuvable: {model_path}, skip.")
        return

    print(f"[core_evaluate:sklearn] Chargement modèle depuis {model_path}")
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    estimator = bundle["estimator"]

    analysis_cfg = params.get("analysis") or {}
    compare_by = analysis_cfg.get("compare_by") or []

    texts, labels_true, rows = load_job_tsv(params)
    texts, labels_true, rows = maybe_debug_subsample_eval(texts, labels_true, params, rows)

    print(f"[core_evaluate:sklearn] Évaluation sur {len(texts)} docs.")
    X = vectorizer.transform(texts)
    labels_pred = estimator.predict(X)

    metrics = compute_basic_metrics(labels_true, list(labels_pred))
    metrics["family"] = "sklearn"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(texts)
    metrics["n_features"] = int(getattr(X, "shape", (0, 0))[1])

    grouped_metrics = compute_grouped_metrics(rows, labels_true, list(labels_pred), compare_by)
    balanced_summary = maybe_attach_balanced_metrics(params, metrics, grouped_metrics)

    save_eval_outputs(params, "sklearn", model_id, metrics)
    save_grouped_metrics(params, "sklearn", model_id, grouped_metrics)
    save_balanced_summary(params, "sklearn", model_id, balanced_summary)


#  Éval HF (squelette)


def eval_hf_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Évaluation générique HuggingFace (famille 'hf').

    - Charge le modèle et le tokenizer depuis models/{corpus}/{view}/hf/{model_id}
    - Évalue sur job.tsv (ou fallback train.tsv)
    - Utilise compute_basic_metrics pour produire metrics.json + meta_eval.json.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[core_evaluate:hf] Transformers ou torch non installés. Skip HF.")
        return

    model_dir = get_model_output_dir(params, "hf", model_id)
    if not model_dir.exists():
        print(f"[core_evaluate:hf] Modèle HF introuvable: {model_dir}, skip.")
        return

    # Charger meta_model pour récupérer le mapping labels
    meta_path = model_dir / "meta_model.json"
    label2id = None
    id2label = None
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        extra = meta.get("extra", {})
        label2id = extra.get("label2id")
        id2label = extra.get("id2label")

    print(f"[core_evaluate:hf] Chargement modèle HF depuis {model_dir}")
    if id2label is not None and label2id is not None and isinstance(id2label, dict):
        # Normaliser les clés (str/int)
        id2label_norm = {int(k): v for k, v in id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            id2label=id2label_norm,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    analysis_cfg = params.get("analysis") or {}
    compare_by = analysis_cfg.get("compare_by") or []

    texts, labels_true, rows = load_job_tsv(params)
    texts, labels_true, rows = maybe_debug_subsample_eval(texts, labels_true, params, rows)

    if not texts:
        print("[core_evaluate:hf] Aucun document à évaluer.")
        return

    class HFEvalDataset(Dataset):
        def __init__(self, texts: List[str], tokenizer, max_length: int):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            return {k: torch.tensor(v) for k, v in enc.items()}

    # max_length : si présent dans meta_model.extra.trainer_params.max_length
    max_length = 256
    if meta:
        extra = meta.get("extra", {})
        max_length = int(extra.get("trainer_params", {}).get("max_length", max_length))

    dataset = HFEvalDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    preds_ids: List[int] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = outputs.logits
            batch_pred = torch.argmax(logits, dim=-1).tolist()
            preds_ids.extend(batch_pred)

    # Conversion ids -> labels
    labels_pred: List[str]
    if id2label and isinstance(id2label, dict):
        id2label_int = {int(k): v for k, v in id2label.items()}
        labels_pred = [id2label_int.get(pid, str(pid)) for pid in preds_ids]
    else:
        # Fallback : labels = str(id)
        labels_pred = [str(pid) for pid in preds_ids]

    metrics = compute_basic_metrics(labels_true, labels_pred)
    metrics["family"] = "hf"
    metrics["model_id"] = model_id
    metrics["n_eval_docs"] = len(texts)

    grouped_metrics = compute_grouped_metrics(rows, labels_true, labels_pred, compare_by)

    balanced_summary = maybe_attach_balanced_metrics(params, metrics, grouped_metrics)

    save_eval_outputs(params, "hf", model_id, metrics)
    save_grouped_metrics(params, "hf", model_id, grouped_metrics)
    save_balanced_summary(params, "hf", model_id, balanced_summary)



#  Éval "check"


def eval_check_model(params: Dict[str, Any], model_id: str = "check_default") -> None:
    """
    Famille 'check' vue comme un pseudo-modèle :
    évaluation = refaire des stats simples sur job.tsv et consigner.
    """
    texts, labels_true, _rows = load_job_tsv(params)
    labels_set = sorted(set(labels_true))
    label_counts = {l: labels_true.count(l) for l in labels_set}

    metrics: Dict[str, Any] = {
        "family": "check",
        "model_id": model_id,
        "n_eval_docs": len(texts),
        "labels": labels_set,
        "label_counts": label_counts,
        "note": "Famille 'check' = pseudo-modèle, évaluation = stats brutes sur job.tsv",
    }

    save_eval_outputs(params, "check", model_id, metrics)

#  main


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    # Seed globale optionnelle (comme dans core_train)
    seed_applied = apply_global_seed(params.get("seed"))
    log("evaluate", "seed", f"Global seed: {'appliquée' if seed_applied else 'non appliquée'} ({params.get('seed')})")


    hw = params.get("hardware", {})
    blas_threads = hw.get("blas_threads", 1)
    set_blas_threads(blas_threads)

    families = params.get("families", []) or []
    if args.only_family and args.only_family in families:
        families = [args.only_family]

    models_to_eval: List[Dict[str, Any]] = []

    if "check" in families:
        models_to_eval.append({"family": "check", "model_id": "check_default"})

    if "spacy" in families:
        for mid in params.get("models_spacy", []) or []:
            models_to_eval.append({"family": "spacy", "model_id": mid})

    if "sklearn" in families:
        for mid in params.get("models_sklearn", []) or []:
            models_to_eval.append({"family": "sklearn", "model_id": mid})

    if "hf" in families:
        for mid in params.get("models_hf", []) or []:
            models_to_eval.append({"family": "hf", "model_id": mid})

    if not models_to_eval:
        print(f"[core_evaluate] Aucun modèle à évaluer pour le profil '{params.get('profile')}'. Rien à faire.")
        return

    print("[core_evaluate] Modèles à évaluer :")
    for m in models_to_eval:
        print(f"  - {m['family']}::{m['model_id']}")

    for m in models_to_eval:
        family = m["family"]
        mid = m["model_id"]
        if family == "spacy":
            eval_spacy_model(params, mid)
        elif family == "sklearn":
            eval_sklearn_model(params, mid)
        elif family == "hf":
            eval_hf_model(params, mid)
        elif family == "check":
            eval_check_model(params, mid)
        else:
            print(f"[core_evaluate] WARNING: famille inconnue '{family}', ignorée.")


if __name__ == "__main__":
    main()

