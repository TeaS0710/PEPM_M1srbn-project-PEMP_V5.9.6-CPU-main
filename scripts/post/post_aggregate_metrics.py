#!/usr/bin/env python
"""
scripts/post/post_aggregate_metrics.py

Parcourt reports/**/metrics.json (+ meta_eval.json) et produit un TSV résumé.

Usage :
    python scripts/post/post_aggregate_metrics.py \
        --reports-dir reports \
        --out reports/summary_all_metrics.tsv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reports-dir",
        default="reports",
        help="Répertoire racine des rapports (défaut: reports).",
    )
    p.add_argument(
        "--out",
        default="reports/summary_all_metrics.tsv",
        help="Chemin du TSV de sortie.",
    )
    return p.parse_args()


def find_metrics_files(reports_dir: Path) -> List[Path]:
    return list(reports_dir.rglob("metrics.json"))


def load_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    metrics_files = find_metrics_files(reports_dir)

    if not metrics_files:
        print(f"[post_aggregate_metrics] Aucun metrics.json trouvé sous {reports_dir}")
        return

    rows: List[Dict[str, str]] = []

    for mpath in metrics_files:
        # On s'attend à une arborescence du type :
        # reports/{corpus_id}/{view}/{family}/{model_id}/metrics.json
        rel = mpath.relative_to(reports_dir)
        parts = rel.parts  # ex: ('web1', 'ideology_global', 'spacy', 'spacy_cnn_quick', 'metrics.json')

        corpus_id = parts[0] if len(parts) > 0 else ""
        view = parts[1] if len(parts) > 1 else ""
        family = parts[2] if len(parts) > 2 else ""
        model_id = parts[3] if len(parts) > 3 else ""

        metrics = load_json(mpath)
        meta_eval_path = mpath.with_name("meta_eval.json")
        meta = load_json(meta_eval_path) if meta_eval_path.exists() else {}

        row = {
            "corpus_id": corpus_id,
            "view": view,
            "family": family,
            "model_id": model_id,
            "profile": meta.get("profile", ""),
            "label_field": meta.get("label_field", ""),
            "pipeline_version": meta.get("pipeline_version", ""),
            "n_eval_docs": str(meta.get("n_eval_docs", "")),
            "accuracy": str(metrics.get("accuracy", "")),
            "macro_f1": str(metrics.get("macro_f1", "")),
            "balanced_accuracy_by_corpus_id": str(metrics.get("balanced_accuracy_by_corpus_id", "")),
            "balanced_macro_f1_by_corpus_id": str(metrics.get("balanced_macro_f1_by_corpus_id", "")),
        }

        # Récupérer quelques infos supplémentaires si dispo
        for k in ("precision_macro", "recall_macro"):
            if k in metrics:
                row[k] = str(metrics[k])

        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Écrire un TSV simple
    if not rows:
        print("[post_aggregate_metrics] Aucun résultat à écrire.")
        return

    # Colonnes dans un ordre stable
    all_keys = [
        "corpus_id",
        "view",
        "family",
        "model_id",
        "profile",
        "label_field",
        "pipeline_version",
        "n_eval_docs",
        "accuracy",
        "macro_f1",
        "balanced_accuracy_by_corpus_id",
        "balanced_macro_f1_by_corpus_id",
        "precision_macro",
        "recall_macro",
    ]
    # S'assurer que les clés existent pour chaque ligne
    for r in rows:
        for k in all_keys:
            r.setdefault(k, "")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(all_keys) + "\n")
        for r in rows:
            f.write("\t".join(r[k] for k in all_keys) + "\n")

    print(f"[post_aggregate_metrics] Écrit {len(rows)} lignes dans {out_path}")


if __name__ == "__main__":
    main()
