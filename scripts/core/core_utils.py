# scripts/core/core_utils.py

import argparse
import os
import time
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional
import pprint as _pprint

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import yaml
import json

console = Console()
COMMON_DIR = os.path.join("configs", "common")
PIPELINE_VERSION = "5.9.6"

#  Utils de base

def load_yaml(path: str) -> Dict[str, Any]:
    """Charger un YAML en dict, avec un message d'erreur clair."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Fusion récursive de dictionnaires (base modifié in-place, retourné)."""
    for k, v in updates.items():
        if (
            isinstance(v, dict)
            and k in base
            and isinstance(base[k], dict)
        ):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_override(raw: str) -> (List[str], Any):
    """
    Parse une override "a.b.c=val" -> (["a","b","c"], "val")
    On laisse la responsabilité de caster au code qui applique
    (int, float, bool, etc.) si besoin.
    """
    if "=" not in raw:
        raise ValueError(f"Override invalide (pas de '='): {raw}")
    key, value = raw.split("=", 1)
    path = key.split(".")
    return path, value

def _smart_cast_override(value: Any) -> Any:
    """
    Caster une valeur d'override (string) vers un type utile :
    - bool, int, float
    - listes de la forme [web1], [web1,asr1], ["web1", "asr1"], []
    - dict JSON éventuel
    Sinon, on retourne la string telle quelle.
    """
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return value

    lower = s.lower()
    # booléens
    if lower in ("true", "false"):
        return lower == "true"
    # none/null
    if lower in ("none", "null"):
        return None

    # int / float
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Listes / dicts entre crochets ou accolades
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []

        # 1) tentative JSON (pour ["web1","asr1"], ["web1", "asr1"], etc.)
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, (list, dict)):
                return parsed
        except Exception:
            pass

        # 2) fallback liste simple : [web1, asr1] -> ["web1","asr1"]
        parts = [p.strip() for p in inner.split(",")]
        items = []
        for p in parts:
            if not p:
                continue
            # enlever guillemets éventuels
            p_clean = p.strip().strip('"').strip("'")
            if not p_clean:
                continue
            items.append(_smart_cast_override(p_clean))
        return items

    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return s



def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Appliquer une liste de 'key=value' sur un dict (nested)."""
    cfg = deepcopy(config)
    for raw in overrides:
        path, value = parse_override(raw)
        cast_val = _smart_cast_override(value)

        # Certains champs sont des listes dans le pipeline (ex: families).
        # Le Makefile fournit souvent une forme compacte: families=spacy,sklearn
        # (sans crochets). On normalise ici pour éviter les itérations caractère
        # par caractère et les profils "Famille inconnue: 'c'".
        list_like_leaf_keys = {
            "families",
            "corpus_ids",
            "models_sklearn",
            "models_spacy",
            "models_hf",
        }
        if path and path[-1] in list_like_leaf_keys and isinstance(cast_val, str):
            cast_val = [p.strip() for p in cast_val.split(",") if p.strip()]

        d: Dict[str, Any] = cfg
        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[path[-1]] = cast_val
    return cfg



def parse_seed(raw, default: Optional[int] = 42) -> Optional[int]:
    """Convertir une seed potentielle en entier fiable.

    Accepte None, "", "none", "null", nombres (int/str). Retourne `default`
    en cas d'échec de conversion.
    """
    if raw is None:
        return default
    if isinstance(raw, str):
        raw_str = raw.strip().lower()
        if raw_str in {"", "none", "null"}:
            return default
        if raw_str in {"time", "now", "auto", "random"}:
            # Seed variable à chaque run (utile pour quick/random sampling).
            return int(time.time())
        try:
            return int(raw_str)
        except Exception:
            return default
    try:
        return int(raw)
    except Exception:
        return default


#  Label maps

def load_label_map(path: str) -> Dict[str, str]:
    """
    Charger un fichier de mapping de labels / acteurs.
    Supporte deux formes :
      1) {'mapping': {clé: valeur, ...}}
      2) {clé: valeur, ...} directement (ex: YAML produit par make_ideology_skeleton
         puis rempli manuellement).
    Retourne un dict {clé: str(valeur_non_vide)}.
    """
    raw = load_yaml(path)
    if "mapping" in raw and isinstance(raw["mapping"], dict):
        mapping = raw["mapping"]
    else:
        mapping = raw

    result: Dict[str, str] = {}
    for k, v in mapping.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        result[str(k)] = v_str
    return result


#  Résolution de profil

def resolve_profile_base(profile_name: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Charge profil + YAML communs (corpora/balance/hardware/models) et construit 'params'.
    Gère les alias (ex: balance_mode -> balance_strategy) et le preset hardware effectif.
    """
    if overrides is None:
        overrides = []

    profile_path = os.path.join("configs", "profiles", f"{profile_name}.yml")
    profile_cfg = load_yaml(profile_path)

    corpora_cfg = load_yaml(os.path.join(COMMON_DIR, "corpora.yml"))
    balance_cfg = load_yaml(os.path.join(COMMON_DIR, "balance.yml"))
    hardware_cfg = load_yaml(os.path.join(COMMON_DIR, "hardware.yml"))
    models_cfg = load_yaml(os.path.join(COMMON_DIR, "models.yml"))

    data_cfg = profile_cfg.get("data") or {}
    analysis_cfg = profile_cfg.get("analysis") or {}

    params: Dict[str, Any] = {
        "profile": profile_cfg.get("profile", profile_name),
        "description": profile_cfg.get("description", ""),
        "profile_raw": profile_cfg,
        "balance_cfg": balance_cfg,
        "hardware_cfg": hardware_cfg,
        "models_cfg": models_cfg,
        "seed": profile_cfg.get("seed", 42),
        "data": deepcopy(data_cfg),
        "analysis": analysis_cfg,
    }

    # Champs simples copiés tels quels si présents
    simple_keys = [
        "corpus_id", "view", "modality",
        "label_field", "label_fields", "label_map",
        "train_prop", "dev_prop", "test_prop", "seed", "min_chars", "max_tokens",
        "tokenizer", "dedup_on", "normalize_mode",
        "generate_spacy_docbin",
        "families",
        "models_spacy", "models_sklearn", "models_hf", "models_check",
        "balance_strategy", "balance_preset", "balance_mode",
        "hardware_preset", "debug_mode",
        "ideology", "actors", "dataset_id",
    ]
    for k in simple_keys:
        if k in profile_cfg:
            params[k] = profile_cfg[k]

    # Construire preset hardware effectif
    hardware_preset  = profile_cfg.get("hardware_preset", "small")
    hardware_presets = hardware_cfg.get("presets", {})
    params["hardware_preset"] = hardware_preset
    params["hardware"] = deepcopy(hardware_presets.get(hardware_preset, {}))

    # Overrides CLI au niveau 'params'
    params["pipeline_version"] = PIPELINE_VERSION
    params = apply_overrides(params, overrides)

    # Re-résoudre le preset hardware après overrides
    hardware_preset_eff = params.get("hardware_preset", profile_cfg.get("hardware_preset", "small"))
    params["hardware_preset"] = hardware_preset_eff
    params["hardware"] = deepcopy(hardware_presets.get(hardware_preset_eff, {}))

    # Propager les limites max_docs top-level dans hardware
    hw = params.setdefault("hardware", {}) or {}
    for k in ("max_train_docs_sklearn", "max_train_docs_spacy", "max_train_docs_hf"):
        if k in params and params[k] is not None:
            hw[k] = params[k]

    # Résolution multi/single corpus
    data_cfg_eff = params.get("data") or data_cfg or {}
    corpus_ids_multi = data_cfg_eff.get("corpus_ids")
    merge_mode = data_cfg_eff.get("merge_mode", "single")
    source_field = data_cfg_eff.get("source_field", "corpus_id")

    corpus_id_single = params.get("corpus_id", profile_cfg.get("corpus_id"))

    if not corpus_ids_multi:
        if not corpus_id_single:
            raise SystemExit(
                f"[config] Profil {profile_name} sans corpus_id ni data.corpus_ids"
            )
        if corpus_id_single not in corpora_cfg:
            raise SystemExit(
                f"[config] corpus_id '{corpus_id_single}' non défini dans common/corpora.yml"
            )

        corpus_cfg = corpora_cfg[corpus_id_single]

        params["corpus_id"] = corpus_id_single
        params["corpus"] = corpus_cfg
        params["corpora"] = [corpus_cfg]
        params["merge_mode"] = "single"
        params["source_field"] = source_field

        dataset_id = params.get("dataset_id") or corpus_id_single
        params["dataset_id"] = dataset_id
    else:
        missing = [cid for cid in corpus_ids_multi if cid not in corpora_cfg]
        if missing:
            raise SystemExit(
                f"[config] corpus_ids inconnus dans common/corpora.yml: {missing}"
            )

        corpora_list = [corpora_cfg[cid] for cid in corpus_ids_multi]
        # IMPORTANT (multi-corpus): ne PAS réutiliser `corpus_id` comme fallback de `dataset_id`.
        # `corpus_id` peut provenir d'un override Makefile (ou d'un ancien workflow) et écrase
        # alors la sortie `data/interim/<dataset_id>/...` en pointant vers un corpus unique.
        # La convention V5.9 :
        #   - dataset_id explicite si fourni
        #   - sinon, fallback stable sur le nom du profil
        dataset_id = params.get("dataset_id") or profile_name

        params["corpora"] = corpora_list
        params["merge_mode"] = merge_mode
        params["source_field"] = source_field
        params["dataset_id"] = dataset_id

        params["corpus_id"] = dataset_id
        params["corpus"] = corpora_list[0]

    # Alias convivial : balance_mode -> balance_strategy
    bm = (params.get("balance_mode") or "").strip().lower()
    if bm == "weights":
        params["balance_strategy"] = "class_weights"
    elif bm == "oversample":
        # par défaut, on utilise cap_docs comme stratégie d'oversampling contrôlable
        params.setdefault("balance_strategy", "cap_docs")

    return params




#  Logging minimal

def log(script: str, stage: str, msg: str) -> None:
    """Log formaté uniforme."""
    print(f"[{script}:{stage}] {msg}")


#  Seed globale optionnelle

def apply_global_seed(seed_val) -> bool:
    """
    Fixe la seed pour random/numpy/torch/spaCy si seed_val est un entier >=0.
    Si seed_val ∈ {None, '', 'none', 'null'} ou <0 -> on n'applique PAS de seed.
    Retourne True si une seed a été appliquée.
    """
    seed = parse_seed(seed_val, default=None)
    if seed is None or seed < 0:
        return False

    import os, random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import spacy.util as spacy_util
        spacy_util.fix_random_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    return True


def debug_print_params(params: Dict[str, Any]) -> None:
    """
    Affiche les paramètres résolus du pipeline de manière lisible.
    - Utilise rich si dispo (tableaux, panneau).
    - Sinon fallback sur pprint.
    """
    if console is None or Table is None or Panel is None:
        print("=== PARAMS RÉSOLUS ===")
        _pprint.pprint(params)
        return

    profile = params.get("profile", "?")
    desc = params.get("description", "")
    version = params.get("pipeline_version", "4.x")

    # Bandeau principal
    header_text = (
        f"[bold]Pipeline {version}[/bold]\n"
        f"[bold]profile=[/bold]{profile}\n"
        f"[bold]version=[/bold]{version}\n"
        f"{desc}"
    )
    console.print()
    console.print(
        Panel.fit(
            header_text,
            title="CONFIG",
            subtitle="params résolus",
            border_style="cyan",
        )
    )

    corpus = params.get("corpus", {})

    # Afficher les *valeurs résolues* (après overrides), pas le profil brut
    families = params.get("families", [])
    models_spacy = params.get("models_spacy", [])
    models_sklearn = params.get("models_sklearn", [])
    models_hf = params.get("models_hf", [])

    def _join_or_str(val: Any) -> str:
        if isinstance(val, str):
            # Autoriser une notation "spacy,sklearn" en string simple
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return ", ".join(parts) if parts else val
        return ", ".join(map(str, val)) if val else "-"

    t1 = Table(title="Corpus & Profil", expand=True)
    t1.add_column("Champ", style="bold", no_wrap=True)
    t1.add_column("Valeur")

    t1.add_row("Profil", str(profile))
    t1.add_row("Desc", desc)
    t1.add_row("Corpus id", str(corpus.get("corpus_id", params.get("corpus_id"))))
    t1.add_row("Corpus path", str(corpus.get("corpus_path", "")))
    t1.add_row("View", str(params.get("view")))
    t1.add_row("Modality", str(params.get("modality")))
    t1.add_row("Label field", str(params.get("label_field")))
    console.print()
    console.print(t1)

    t2 = Table(title="Familles & Modèles", expand=True)
    t2.add_column("Famille", style="bold", no_wrap=True)
    t2.add_column("Valeur")

    t2.add_row("Families", _join_or_str(families))
    t2.add_row("spaCy", _join_or_str(models_spacy))
    t2.add_row("sklearn", _join_or_str(models_sklearn))
    t2.add_row("HF", _join_or_str(models_hf))
    console.print()
    console.print(t2)

    hw = params.get("hardware", {})
    t3 = Table(title="Balance & Hardware", expand=True)
    t3.add_column("Champ", style="bold", no_wrap=True)
    t3.add_column("Valeur")

    t3.add_row("Balance strategy", str(params.get("balance_strategy")))
    t3.add_row("Balance preset", str(params.get("balance_preset")))
    t3.add_row("HW preset", str(params.get("hardware_preset")))
    t3.add_row("RAM (GB)", str(hw.get("ram_gb", "?")))
    t3.add_row("Max procs", str(hw.get("max_procs", "?")))
    t3.add_row("TSV chunk rows", str(hw.get("tsv_chunk_rows", "?")))
    t3.add_row("spaCy shard docs", str(hw.get("spacy_shard_docs", "?")))
    console.print()
    console.print(t3)

    t4 = Table(title="Train & Divers", expand=True)
    t4.add_column("Champ", style="bold", no_wrap=True)
    t4.add_column("Valeur")

    t4.add_row("Train prop", str(params.get("train_prop")))
    t4.add_row("Min chars", str(params.get("min_chars")))
    t4.add_row("Max tokens", str(params.get("max_tokens")))
    t4.add_row("Tokenizer", str(params.get("tokenizer")))
    t4.add_row("Seed", str(params.get("seed")))
    t4.add_row("Debug mode", str(params.get("debug_mode")))
    console.print()
    console.print(t4)
    console.print()

