# PEPM – Projet Étude Politique Master (Pipeline PEMP)

Version: **5.9.6** (CPU-min)

Ce dépôt contient une version **réduite** du pipeline PEMP utilisée pour PEPM :
- préparation de datasets depuis un `teiCorpus` (TEI-XML) ;
- entraînement de modèles **sklearn** (baseline TF‑IDF + classifieurs) ;
- évaluation et génération de rapports (`metrics.json`, `classification_report.txt`, etc.).

Objectif de la réduction : garder **l’essentiel pour reproduire les analyses** (idéologie binaire + classification “crawl” multi-sites), en supprimant le code expérimental et les systèmes de vérification périphériques.

## Prérequis
- Python 3.11+ (testé avec Python 3.12)
- Données TEI dans `data/raw/<corpus_id>/corpus.xml` (non incluses ici)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> Notes :
> - `requirements-full.txt` correspond à l’environnement complet historique (spaCy/HF). Cette version 5.9.6 se limite volontairement à sklearn.

## Commandes principales

### 0) Informations machine (optionnel)
```bash
make sysinfo
```

### 1) Idéologie binaire (web1) – baseline
```bash
make pipeline PROFILE=ideo_quick
```

### 2) Idéologie binaire (mix web1 + asr1)
```bash
make pipeline PROFILE=ideo_quick_web1_asr1 DATASET_ID=FINAL_mix_ideo
```

### 3) Classification “crawl” (20 sites) – cap docs / label
```bash
make pipeline PROFILE=crawl_quick_web1 DATASET_ID=FINAL_web1_crawl
```

### 4) Audit (optionnel)
Audit TEI :
```bash
make audit_tei CORPUS_ID=web1
```

Audit des splits (TSV) :
```bash
make audit_splits DATASET_ID=FINAL_web1_crawl VIEW=crawl
```

### 5) Nettoyage
```bash
make clean
```

## Arborescence minimale

- `configs/`
  - `common/` : configuration commune (corpora, hardware, balance, modèles)
  - `profiles/` : profils d’exécution prêts à l’emploi
  - `label_maps/` : mappings d’acteurs / idéologies
- `scripts/core/` : préparation / entraînement / évaluation
- `scripts/tools/` : audits et sysinfo
- `scripts/post/` : agrégation des métriques

## Sorties

- Datasets : `data/interim/<DATASET_ID>/<view>/{train,dev,test}.tsv`
- Modèles : `models/<DATASET_ID>/<view>/sklearn/<model>/model.joblib`
- Rapports : `reports/<DATASET_ID>/<view>/sklearn/<model>/*`

