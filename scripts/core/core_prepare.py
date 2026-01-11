# scripts/core/core_prepare.py

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

# Très gros corpus TSV/TEI : sécuriser la taille maximale de champ CSV.
try:
    csv.field_size_limit(2**31 - 1)
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Normalisation / compat
ASR_PARTY_FALLBACK = True

# Runtime normalisation flags (param-driven)
DROP_SEGMENTS = False
DROP_SEGMENTS_MODALITIES = set()  # empty => apply to all
REDACT_META_TOKENS = False


import xml.etree.ElementTree as ET

from scripts.core.core_utils import (
    resolve_profile_base,
    load_label_map,
    load_yaml,
    debug_print_params,
    PIPELINE_VERSION,
    log,
    parse_seed,
)
import yaml

# Cache des label_maps chargés (mapping + unknown_labels)
_LABEL_MAP_CACHE: Dict[str, Dict[str, Any]] = {}


@lru_cache(maxsize=1)
def load_ideology_actors(path: str) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Charger le mapping acteurs depuis un YAML unique (crawls + domaines)."""

    if not path:
        return {}, {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    actors = data.get("actors")
    if not isinstance(actors, dict):
        actors = {}

    crawl_to_actor: Dict[str, str] = {}
    domain_to_actor: Dict[str, str] = {}
    for actor_id, info in actors.items():
        for crawl_id in (info or {}).get("crawls", []) or []:
            crawl_norm = normalize_label_value(str(crawl_id))
            if crawl_norm:
                crawl_to_actor[crawl_norm] = actor_id
        for domain_id in (info or {}).get("domains", []) or []:
            domain_norm = normalize_label_value(str(domain_id))
            if domain_norm:
                domain_to_actor[domain_norm] = actor_id
    return actors, crawl_to_actor, domain_to_actor


# CLI


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V5.9 core_prepare : TEI -> TSV -> formats (multi-famille)"
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
        help="Override config (clé=valeur, ex: train_prop=0.7, corpus_id=web2)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne produit pas les fichiers, imprime seulement les stats de la vue",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# Tokenizers & poids de classe


def _tokenize_whitespace(text: str) -> List[str]:
    """Tokenisation naïve sur espaces (comportement historique)."""
    return text.split()


def _tokenize_simple(text: str) -> List[str]:
    """
    Tokenisation simple : sépare les mots alphanumériques et la ponctuation.
    Utile pour comparer l'effet d'un tokeniseur plus fin.
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _lazy_spacy(lang_code: str):
    try:
        import spacy
    except ImportError:
        raise SystemExit("[core_prepare] Tokenizer 'spacy' demandé mais spaCy n'est pas installé.")
    try:
        return spacy.blank(lang_code)
    except Exception:
        # fallback grossier
        return spacy.blank("xx")

_SPACY_NLP_CACHE = {}

def _get_spacy_nlp(lang: str):
    # cache pour éviter de recharger
    nlp = _SPACY_NLP_CACHE.get(lang)
    if nlp is None:
        try:
            import spacy
            nlp = spacy.blank(lang)
        except Exception:
            nlp = None
        _SPACY_NLP_CACHE[lang] = nlp
    return nlp

def count_tokens(text: str, tokenizer_name: str) -> int:
    """
    tokenizer_name:
      - 'split' / 'whitespace' -> text.split() (tokenisation espaces)
      - 'simple'               -> segmentation simple (fallback historique)
      - 'spacy:<lang>'         -> ex: 'spacy:fr', 'spacy:xx'
    """
    name = (tokenizer_name or "split").strip().lower()
    if name == "simple":
        return len([t for t in text.replace("\n", " ").split(" ") if t])
    if name in {"split", "whitespace"}:
        return len(text.split())
    if name.startswith("spacy:"):
        lang = name.split(":", 1)[1] or "xx"
        nlp = _lazy_spacy(lang)
        return len([t for t in nlp.make_doc(text)])
    # défaut
    return len(text.split())

def _norm_key_for_dedup(s: str) -> str:
    """Normaliser un texte pour déduplication / anti-fuite.

    Objectif: aligner la clé sur l'audit (réduction de bruit: casse, ponctuation,
    espaces). Ce n'est pas une normalisation linguistique, juste une clé robuste
    pour identifier des doublons quasi-exacts qui créent de la fuite train/test.
    """
    s = (s or "").lower()
    # Remplacer la ponctuation par des espaces (conservatif pour l'audit).
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = " ".join(s.split()).strip()
    return s


def _text_hash_strong(text: str) -> str:
    """Hash stable du texte normalisé (anti-fuite cross-splits)."""
    key = _norm_key_for_dedup(text)
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


def _drop_overlap_by_text(
    splits: Dict[str, List[Dict[str, Any]]],
    order: List[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Supprime les doublons entre splits (priorité: ordre donné).

    Politique volontairement simple et sûre:
      - si un texte (normalisé) apparaît déjà dans un split précédent, on le retire
        des splits suivants.
      - idem pour les IDs, si un même id est présent plusieurs fois.

    Objectif: garantir 0 fuite pour l'audit (overlap id/text_hash), sans
    complexifier la stratification.
    """
    seen_text = set()
    seen_ids = set()
    removed: Dict[str, int] = {k: 0 for k in order}
    out: Dict[str, List[Dict[str, Any]]] = {}

    for split_name in order:
        bucket = splits.get(split_name) or []
        kept: List[Dict[str, Any]] = []
        for d in bucket:
            # 1) anti-overlap id
            rid = (d.get("id") or d.get("xml_id") or "").strip()
            if rid and rid in seen_ids:
                removed[split_name] += 1
                continue

            # 2) anti-overlap texte
            h = _text_hash_strong(d.get("text") or "")
            if h in seen_text:
                removed[split_name] += 1
                continue

            if rid:
                seen_ids.add(rid)
            seen_text.add(h)
            kept.append(d)

        out[split_name] = kept

    return out, removed


def get_default_spacy_lang(params: Dict[str, Any]) -> str:
    """Choisir une langue par défaut pour la construction des DocBin spaCy.

    On essaie d'abord de regarder dans models.yml (famille spacy),
    via les modèles référencés par le profil. Sinon, on tombe sur 'fr'.
    """
    families_cfg = params.get("models_cfg", {}).get("families", {})
    spacy_models = families_cfg.get("spacy", {})
    for mid in (params.get("models_spacy") or []):
        cfg = spacy_models.get(mid)
        if cfg and cfg.get("lang"):
            return cfg["lang"]
    return "fr"



def compute_class_weights_from_counts(label_counts: Counter) -> Dict[str, float]:
    """
    Calculer des poids de classe de type 'balanced' (sklearn) :

        w(label) = n_samples / (n_labels * n_label)

    Ne modifie pas les docs, mais donne des poids utilisables
    par certains modèles (sklearn, HF, etc.).
    """
    total = sum(label_counts.values())
    n_labels = len(label_counts) or 1
    if total <= 0:
        return {lab: 1.0 for lab in label_counts}

    weights: Dict[str, float] = {}
    for lab, c in label_counts.items():
        if c <= 0:
            weights[lab] = 0.0
        else:
            weights[lab] = total / (n_labels * c)
    return weights



# TEI helpers

def _token_count(s: str) -> int:
    """Compter grossièrement les tokens comme dans V2 (split() whitespace)."""
    return len(s.split()) if s else 0


def _local_name(tag: str) -> str:
    """Extraire le nom local d'un tag XML (sans namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def iter_tei_docs(tei_path: str) -> Iterable[ET.Element]:
    """
    Itérer sur les documents TEI en considérant chaque élément <TEI> comme
    un document complet (teiHeader + text).

    Cette approche est alignée sur les corpus fournis (racine <teiCorpus>
    contenant plusieurs <TEI>), ce qui garantit que les métadonnées du
    header (label, modality, id…) restent accessibles au moment de
    l'extraction.
    """
    # IMPORTANT (gros corpus): avec ElementTree, ``elem.clear()`` seul peut
    # laisser des "coquilles" d'éléments accrochées à la racine (<teiCorpus>),
    # ce qui fait grossir la mémoire au fil du parsing. On utilise donc le
    # pattern iterparse start/end pour récupérer la racine et retirer les
    # noeuds <TEI> une fois consommés.
    context = ET.iterparse(tei_path, events=("start", "end"))
    _event0, root = next(context)  # racine du document
    for event, elem in context:
        if event == "end" and _local_name(elem.tag) == "TEI":
            yield elem
            # À la reprise du générateur (après consommation côté appelant)
            # on détache le noeud TEI de la racine pour libérer la mémoire.
            try:
                root.remove(elem)
            except Exception:
                # Fallback: au minimum, on vide le sous-arbre.
                elem.clear()



def drop_segments_inplace(tei: ET.Element) -> int:
    """Supprime les blocs <ab type="segments"> (souvent redondants avec le texte complet).

    Retourne le nombre de noeuds supprimés.
    """
    removed = 0

    def _recurse(parent: ET.Element) -> None:
        nonlocal removed
        for child in list(parent):
            if _local_name(child.tag) == "ab" and (child.get("type") or "").strip().lower() == "segments":
                parent.remove(child)
                removed += 1
                continue
            _recurse(child)

    _recurse(tei)
    return removed


def itertext_excluding(elem: ET.Element, exclude_pred):
    """Recursive itertext with subtree exclusion.

    ElementTree has no parent pointers; this helper allows skipping subtrees
    (e.g. ASR <ab type="segments"> blocks) to avoid text duplication.
    """
    if exclude_pred(elem):
        return
    if elem.text:
        yield elem.text
    for child in list(elem):
        yield from itertext_excluding(child, exclude_pred)
        if child.tail:
            yield child.tail


def extract_term(elem: ET.Element, term_type: str) -> Optional[str]:
    """
    Extraire le texte d'un <term type="xxx"> dans l'élément ou ses descendants.

    Utilisé pour les labels (ideology, crawl, ...) et pour modality.
    """
    for t in elem.iter():
        if _local_name(t.tag) == "term" and t.get("type") == term_type:
            if t.text:
                txt = t.text.strip()
                if txt:
                    return txt

    # Fallback ASR : certains corpus encodent la "source" via <classCode scheme="asr_party">.
    # On expose alors une pseudo-valeur compatible avec les label_maps ("asr_party:<X>").
    if ASR_PARTY_FALLBACK and term_type in ("domain", "crawl"):
        for cc in elem.iter():
            if _local_name(cc.tag) == "classCode" and (cc.get("scheme") or "").strip() == "asr_party":
                v = (cc.text or "").strip()
                if v:
                    return f"asr_party:{v}"

    return None


def normalize_label_value(value: str) -> str:
    """
    Normaliser une valeur de label pour matcher des clés de mapping,
    indépendamment des séparateurs (espaces, tirets, slash).

    Exemples :
      "crawl-actionfrancaise-20251003_000000" → "crawl_actionfrancaise_20251003_000000"
      "Far Left" → "far_left"
    """
    norm = value.strip().lower()
    norm = re.sub(r"[\s/\\]+", "_", norm)
    norm = re.sub(r"[^0-9a-zA-Z_]+", "_", norm)
    norm = re.sub(r"_+", "_", norm)
    return norm.strip("_")


def _load_label_map_cached(path: Optional[str]) -> Dict[str, Any]:
    """Charger un label_map en normalisant les clés et en respectant unknown_labels."""
    if not path:
        return {"mapping": {}, "unknown_labels": {}}
    if path in _LABEL_MAP_CACHE:
        return _LABEL_MAP_CACHE[path]

    raw = load_label_map(path)
    norm_map = {normalize_label_value(str(k)): v for k, v in raw.items()}

    unknown_cfg: Dict[str, Any] = {}
    try:
        full_raw = load_yaml(path)
        if isinstance(full_raw, dict) and isinstance(full_raw.get("unknown_labels"), dict):
            unknown_cfg = full_raw["unknown_labels"]
    except Exception:
        unknown_cfg = {}

    _LABEL_MAP_CACHE[path] = {"mapping": norm_map, "unknown_labels": unknown_cfg}
    return _LABEL_MAP_CACHE[path]


def apply_label_mapping(
    raw_value: str, mapping: Dict[str, str], unknown_cfg: Dict[str, Any]
) -> Optional[str]:
    """
    Appliquer un mapping de labels en respectant une politique unknown_labels.

    unknown_cfg.policy : drop (None), keep (valeur normalisée), other (other_label).
    """

    raw_norm = normalize_label_value(raw_value)
    mapped = mapping.get(raw_norm)
    if mapped:
        return mapped

    policy = str(unknown_cfg.get("policy", "drop")).strip().lower()
    if policy == "keep":
        return raw_norm
    if policy == "other":
        return str(unknown_cfg.get("other_label", "other"))
    return None


def stratified_split(
    docs: List[Dict[str, Any]], train_prop: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Découpage stratifié (par label) avant équilibrage."""

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_label[d["label"]].append(d)

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    job: List[Dict[str, Any]] = []
    for lab, bucket in by_label.items():
        rng.shuffle(bucket)
        n_total = len(bucket)
        n_train = int(round(train_prop * n_total))
        train.extend(bucket[:n_train])
        job.extend(bucket[n_train:])

    rng.shuffle(train)
    rng.shuffle(job)
    return train, job


def resolve_ideology_label(
    row_meta: Dict[str, Any], ideology_cfg: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], str]:
    """Résoudre le label idéologique final selon la config structurée."""

    return resolve_ideology_label_actors(row_meta, ideology_cfg)


def resolve_ideology_label_actors(
    row_meta: Dict[str, Any], ideology_cfg: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], str]:
    actors_yaml = ideology_cfg.get("actors_yaml")
    actors_table, crawl_to_actor, domain_to_actor = load_ideology_actors(actors_yaml)

    crawl_norm = normalize_label_value(str(row_meta.get("crawl") or ""))
    domain_norm = normalize_label_value(str(row_meta.get("domain") or ""))
    actor_id = None

    if domain_norm:
        actor_id = domain_to_actor.get(domain_norm)
    if not actor_id:
        actor_id = crawl_to_actor.get(crawl_norm)

    if not actor_id:
        policy = str(ideology_cfg.get("unknown_actors", {}).get("policy", "drop")).strip().lower()
        if policy == "keep":
            unknown_label = str(ideology_cfg.get("unknown_actors", {}).get("label", "unknown_actor"))
            return None, unknown_label, "unknown_actor_kept"
        return None, None, "unknown_actor_dropped"

    view = ideology_cfg.get("view", "binary")
    actor_info = actors_table.get(actor_id, {}) or {}
    label = None

    if view == "binary":
        label = actor_info.get("side_binary")
    elif view == "five_way":
        label = actor_info.get("global_five")
    elif view == "left_intra":
        if actor_info.get("side_binary") != "left":
            return actor_id, None, "actor_not_left"
        label = actor_info.get("intra_left")
    elif view == "right_intra":
        if actor_info.get("side_binary") != "right":
            return actor_id, None, "actor_not_right"
        label = actor_info.get("intra_right")
    else:
        return actor_id, None, f"unknown_view_{view}"

    if not label:
        return actor_id, None, "label_missing_for_view"
    return actor_id, str(label), "ok"


def extract_doc_id(elem: ET.Element, fallback_idx: int) -> str:
    """
    Essayer de récupérer un identifiant de document dans les attributs TEI,
    sinon utiliser un index numérique.
    """
    # Exemples possibles: @xml:id sur <TEI>, @n, etc.
    xml_id = elem.get("{http://www.w3.org/XML/1998/namespace}id") or elem.get("xml:id")
    if xml_id:
        return str(xml_id)
    n_attr = elem.get("n")
    if n_attr:
        return str(n_attr)
    return f"doc_{fallback_idx}"


def normalize_text(text: str, mode: str = "none") -> str:
    """Normalisation optionnelle du texte pour réduire certains biais de surface.

    - none: pas de normalisation (texte brut)
    - basic: Unicode NFKC + minuscules + espaces normalisés
    - aggressive: basic + suppression des diacritiques
    - debias: aggressive + suppression grossière d'URLs / emails (avant redaction meta)
    """
    import re as _re
    import unicodedata as _ud

    if text is None:
        return ""
    s = str(text)

    mode = (mode or "none").strip().lower()
    if mode in ("none", "off", "false", "0", ""):
        return s

    # Unicode normalization + common whitespace cleanup
    s = _ud.normalize("NFKC", s)
    s = s.replace(" ", " ")  # nbsp
    s = _re.sub(r"\s+", " ", s).strip()

    if mode in ("basic", "light", "std", "standard"):
        return s.lower()

    if mode in ("aggressive", "full", "strong", "debias"):
        s = s.lower()
        # strip accents/diacritics
        s = _ud.normalize("NFD", s)
        s = "".join(ch for ch in s if _ud.category(ch) != "Mn")
        s = _ud.normalize("NFKC", s)
        s = _re.sub(r"\s+", " ", s).strip()

        if mode == "debias":
            # remove obvious URLs / emails (coarse)
            s = _re.sub(r"https?://\S+", " ", s)
            s = _re.sub(r"www\.\S+", " ", s)
            s = _re.sub(r"\b\S+@\S+\b", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
        return s

    # Unknown mode -> safe fallback
    return s

def extract_text(tei: ET.Element, drop_segments: bool = False) -> str:
    """Extraire le texte principal d'un TEI (corps) en conservant l'ordre.

    Stratégie:
      - repérer <text> (ou fallback TEI),
      - concaténer le contenu texte en préservant l'ordre,
      - (optionnel) exclure les blocs ASR segmentés (<ab type="segments">) pour éviter les duplications,
      - normaliser les espaces.

    La normalisation 'méthodologique' (none/basic/aggressive/debias) est appliquée plus tard
    via normalize_text(...), puis (optionnel) via redact_meta_tokens(...).
    """
    parts = []

    def _exclude_pred(el: ET.Element) -> bool:
        if not drop_segments:
            return False
        try:
            tag = el.tag.split('}', 1)[-1]
        except Exception:
            tag = str(el.tag)
        if tag != 'ab':
            return False
        # TEI: <ab type="segments">...</ab>
        t = (el.get('type') or '').strip().lower()
        return t == 'segments'

    try:
        target = None
        for node in tei.iter():
            tag = node.tag.split('}', 1)[-1]
            if tag == 'text':
                target = node
                break
        if target is None:
            target = tei

        for chunk in itertext_excluding(target, _exclude_pred):
            if chunk and chunk.strip():
                parts.append(chunk.strip())
    except Exception:
        # Fallback ultra robuste
        try:
            if getattr(tei, 'text', None):
                parts.append(str(tei.text))
        except Exception:
            pass

    text = ' '.join(parts)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def redact_meta_tokens(text: str, meta: dict) -> str:
    """Supprimer du texte certaines chaînes issues des métadonnées.

    Objectif: limiter les fuites (leakage) lorsque des tokens comme le nom de domaine
    ou l'identifiant de crawl apparaissent dans le contenu.

    Heuristiques conservatrices:
      - on supprime seulement des tokens "spécifiques" (contiennent '.' ou ':' ou commencent par 'crawl')
        ou de longueur >= 8.
      - on applique une substitution insensible à la casse.
    """
    import re as _re

    if not text:
        return ''
    s = str(text)
    toks = []
    for key in ('domain', 'crawl'):
        v = (meta or {}).get(key)
        if not v:
            continue
        v = str(v).strip()
        if not v:
            continue
        vv = v.lower()
        specific = ('.' in vv) or (':' in vv) or vv.startswith('crawl') or (len(vv) >= 8)
        if not specific:
            continue
        toks.append(v)
        if vv.startswith('www.'):
            toks.append(v[4:])

    # also remove 'asr_party:xxx' if present
    v = (meta or {}).get('domain')
    if v and str(v).lower().startswith('asr_party:'):
        toks.append(str(v))

    for tok in sorted(set(toks), key=len, reverse=True):
        try:
            s = _re.sub(_re.escape(tok), ' ', s, flags=_re.IGNORECASE)
        except Exception:
            pass

    s = _re.sub(r'\s+', ' ', s).strip()
    return s

def extract_modality(elem: ET.Element, params: Dict[str, Any]) -> str:
    """
    Extraire la modalité du document :
      1) d'abord à partir de <term type="modality">...,
      2) sinon via default_modality du corpus,
      3) sinon 'unknown'.
    """
    term_mod = extract_term(elem, "modality")
    if term_mod:
        return term_mod

    corpus_default = params.get("corpus", {}).get("default_modality")
    if corpus_default:
        return corpus_default

    return "unknown"


def extract_label_raw(elem: ET.Element, label_field: Optional[Any]) -> Optional[str]:
    """
    Extraire le label brut à partir du TEI, en fonction de label_field.
    Par défaut, on cherche un <term type='label_field'>.

    Ex:
      label_field = "ideology" -> <term type="ideology">gauche</term>
      label_field = "crawl"    -> <term type="crawl">X</term>
    """
    if not label_field:
        return None

    if isinstance(label_field, (list, tuple)):
        fields = list(label_field)
    else:
        fields = [label_field]

    for field in fields:
        val = extract_term(elem, field)
        if val:
            return val
    return None


# Vue : TEI -> TSV


def build_view(params: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    normalize_cfg = params.get("normalize") or {}

    # runtime flags
    global ASR_PARTY_FALLBACK, DROP_SEGMENTS, DROP_SEGMENTS_MODALITIES, REDACT_META_TOKENS
    ASR_PARTY_FALLBACK = bool(normalize_cfg.get("asr_party_fallback", True))
    DROP_SEGMENTS = bool(normalize_cfg.get("drop_segments", False))
    DROP_SEGMENTS_MODALITIES = set(normalize_cfg.get("drop_segments_modalities", []) or [])
    REDACT_META_TOKENS = bool(normalize_cfg.get("redact_meta_tokens", False))

    # mode texte
    normalize_mode = (params.get("normalize_mode") or normalize_cfg.get("mode") or ("basic" if normalize_cfg else "none"))
    """
    Construire la vue supervisée (train.tsv, job.tsv, meta_view.json).
    Retourne un dict de meta pour usage ultérieur (log ou tests).
    """
    corpus_single = params.get("corpus")
    corpora_multi = params.get("corpora") or ([corpus_single] if corpus_single else [])
    source_field = params.get("source_field", "corpus_id")
    merge_mode = params.get("merge_mode", "single")
    label_fields = params.get("label_fields")
    label_field = params.get("label_field")
    # Priorité à label_fields (liste) si présent, sinon label_field (héritage)
    fields_for_labels = label_fields if label_fields is not None else label_field
    label_map_path = params.get("label_map")
    label_map_data = _load_label_map_cached(label_map_path)
    label_map = label_map_data.get("mapping")
    label_map_unknown_cfg = label_map_data.get("unknown_labels", {})

    ideology_cfg = params.get("ideology") or {}
    actors_cfg = params.get("actors") or {}

    modality_filter = params.get("modality")  # ex: "web", "asr", etc.
    train_prop = float(params.get("train_prop", 0.8))
    min_chars = params.get("min_chars", 0) or 0
    max_tokens = params.get("max_tokens")  # peut rester None
    tokenizer_name = params.get("tokenizer", "split")

    balance_strategy = params.get("balance_strategy", "none")
    balance_preset = params.get("balance_preset")

    corpus_id = params.get("corpus_id", (corpus_single or {}).get("corpus_id", "unknown_corpus"))
    dataset_id = params.get("dataset_id") or corpus_id
    view = params.get("view", "unknown_view")

    interim_dir = os.path.join("data", "interim", str(dataset_id), view)
    os.makedirs(interim_dir, exist_ok=True)

    # Collecte des docs (RAM V1 ; optimisable ensuite en streaming/sharding)
    docs: List[Dict[str, Any]] = []
    modality_counts = Counter()
    label_counts = Counter()

    meta_fields: set = set()
    if ideology_cfg:
        meta_fields.update({"crawl", "domain"})
    if actors_cfg:
        meta_fields.add("actor")
    ideology_stats: Counter = Counter()
    idx_global = 0
    for corpus in corpora_multi:
        tei_path = corpus["corpus_path"]
        corpus_id_src = corpus.get("corpus_id", "unknown_corpus")
        print(f"[core_prepare] Lecture TEI: {tei_path} (source={corpus_id_src})")
        for elem in iter_tei_docs(tei_path):
            idx_global += 1
            # xml_id_raw est l'identifiant TEI au sein du corpus courant.
            # En multi-corpus, on le namespace pour garantir l'unicité globale et
            # éviter des collisions d'IDs (web1 et asr1 ont souvent les mêmes doc_000001...).
            xml_id_raw = extract_doc_id(elem, idx_global)
            doc_id = xml_id_raw
            if len(corpora_multi) > 1 and not str(doc_id).startswith(f"{corpus_id_src}__"):
                doc_id = f"{corpus_id_src}__{doc_id}"

            # Modalité (avec fallback default_modality du corpus courant)
            doc_modality = extract_modality(elem, {**params, 'corpus': corpus})
            modality_counts[doc_modality] += 1
            if modality_filter and doc_modality != modality_filter:
                continue

            # Métadonnées (domain/crawl/actor/...)
            row_meta = {field: extract_term(elem, field) for field in meta_fields}
            row_meta[source_field] = corpus_id_src

            # Extraction texte (option: suppression des segments ASR redondants)
            drop_seg = bool(DROP_SEGMENTS) and (not DROP_SEGMENTS_MODALITIES or doc_modality in DROP_SEGMENTS_MODALITIES)
            text_raw = extract_text(elem, drop_segments=drop_seg)
            text = normalize_text(text_raw, normalize_mode)
            if REDACT_META_TOKENS or str(normalize_mode).lower() == 'debias':
                text = redact_meta_tokens(text, row_meta)

            if len(text) < min_chars:
                continue

            # compter les tokens selon le tokenizer configuré
            tokens = count_tokens(text, tokenizer_name)

            # Filtre max_tokens basé sur ce tokenizer
            if max_tokens and tokens > max_tokens:
                continue


            label_raw: Optional[str] = None
            label: Optional[str] = None

            if ideology_cfg:
                label_raw, label, ideology_reason = resolve_ideology_label(row_meta, ideology_cfg)
                if ideology_reason and ideology_reason != "ok":
                    ideology_stats[ideology_reason] += 1
            else:
                # Cas non-idéologie (ex: vue "crawl", "domain", etc.)
                # 1) On récupère le label brut dans le TEI
                label_raw = extract_label_raw(elem, fields_for_labels)
                if not label_raw:
                    continue

                if label_map_path:
                    # Un label_map est configuré : on applique le mapping
                    mapped = apply_label_mapping(
                        label_raw,
                        label_map or {},
                        label_map_unknown_cfg,
                    )
                    if not mapped:
                        # Droppé par la policy unknown_labels
                        continue
                    label = mapped
                else:
                    # Aucun label_map -> on garde le label brut (normalisé) tel quel
                    # Comportement "historique" : classification directe par crawl/domain/etc.
                    label = normalize_label_value(label_raw)

            if not label:
                continue


            label_counts[label] += 1
            docs.append(
                {
                    "id": doc_id,
                    "xml_id": xml_id_raw,
                    "label": label,
                    "label_raw": label_raw,
                    "text": text,
                    "modality": doc_modality,
                    "tokens": tokens,  # pour cap_tokens/stats/etc.
                    "meta": row_meta,
                }
            )

    if not docs:
        raise SystemExit("[core_prepare] Aucun document valide après filtrage. Vérifie ton profil/config.")

    print(f"[core_prepare] Docs retenus avant équilibrage: {len(docs)}")
    print(f"[core_prepare] Répartition labels (avant balance): {label_counts}")
    print(f"[core_prepare] Répartition modalités (tous docs vus): {modality_counts}")

    # --- Déduplication optionnelle (par doc), AVANT équilibrage ---

    log("prepare", "collect", f"Docs retenus avant dedup: {len(docs)}")
    label_counts_raw = Counter(d["label"] for d in docs)

    dedup_on = str(params.get("dedup_on") or "none").strip().lower()
    if dedup_on not in {"none", "", "no", "id", "text", "id_and_text"}:
        log("prepare", "dedup", f"Valeur inconnue '{dedup_on}', fallback none")
        dedup_on = "none"

    if dedup_on != "none":
        seen = set()
        deds = []
        for d in docs:
            if dedup_on == "id":
                key = (d.get("id") or d.get("xml_id") or "").strip()
            elif dedup_on == "text":
                key = _norm_key_for_dedup(d.get("text") or "")
            else:  # id_and_text
                key = (d.get("id") or d.get("xml_id") or "").strip() + "||" + _norm_key_for_dedup(d.get("text") or "")
            if key in seen:
                continue
            seen.add(key)
            deds.append(d)
        log("prepare", "dedup", f"Dedup '{dedup_on}': {len(docs)} -> {len(deds)}")
        docs = deds

    # Recompter après dedup
    label_counts = Counter(d["label"] for d in docs)

    # Seed résolue une fois (supporte 'time' dans les profils quick)
    seed = parse_seed(params.get("seed"), default=42) or 42
    params["seed_resolved"] = seed


    # Limite globale sur le nombre de documents (debug / smoke tests)
    max_docs_global = int(params.get("max_docs_global") or 0)
    max_docs_global_mode = str(params.get("max_docs_global_mode") or "head").strip().lower()
    if max_docs_global_mode in {"", "none", "null"}:
        max_docs_global_mode = "head"
    if max_docs_global_mode not in {"head", "first", "random", "shuffle"}:
        log("prepare", "limit", f"max_docs_global_mode inconnu '{max_docs_global_mode}' → fallback 'head'")
        max_docs_global_mode = "head"

    if max_docs_global > 0 and len(docs) > max_docs_global:
        if max_docs_global_mode in {"random", "shuffle"}:
            # Echantillonnage sans remplacement, piloté par la seed (reproductible si seed fixe).
            rng = random.Random(seed)
            docs = rng.sample(docs, k=max_docs_global)
        else:
            # Comportement historique : on prend les premiers documents (ordre TEI).
            docs = docs[:max_docs_global]

        label_counts = Counter(d["label"] for d in docs)
        log(
            "prepare",
            "limit",
            f"max_docs_global={max_docs_global} ({max_docs_global_mode}) applique : {len(docs)} docs conserves apres dedup",
        )




    # Filtrage éventuel par acteurs
    actor_counts_before = Counter()
    actor_counts_after = Counter()
    if actors_cfg:
        include = set(actors_cfg.get("include") or [])
        min_docs = int(actors_cfg.get("min_docs", 0) or 0)

        for d in docs:
            actor_counts_before[(d.get("meta") or {}).get("actor") or "unknown"] += 1

        if include:
            docs = [d for d in docs if (d.get("meta") or {}).get("actor") in include]

        if min_docs > 0:
            by_actor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for d in docs:
                by_actor[(d.get("meta") or {}).get("actor")].append(d)
            docs = [doc for actor, bucket in by_actor.items() if len(bucket) >= min_docs for doc in bucket]

        for d in docs:
            actor_counts_after[(d.get("meta") or {}).get("actor") or "unknown"] += 1

        if include or min_docs > 0:
            print(
                f"[core_prepare] Filtrage acteurs -> {len(actor_counts_after)} acteurs conservés, {len(docs)} docs"
            )

    # Split stratifié AVANT équilibrage (seed déjà résolue plus haut)

    # --- split 3 parties (train/dev/test) ---
    train_prop = float(params.get("train_prop", train_prop))
    dev_prop = params.get("dev_prop")
    test_prop = params.get("test_prop")

    # First split: train vs rest
    train_docs_raw, rest_docs = stratified_split(docs, train_prop, seed)

    # Second split: dev vs test (inside rest)
    if dev_prop is None and test_prop is None:
        # default: split the remaining equally
        dev_prop = (1.0 - train_prop) / 2.0
        test_prop = (1.0 - train_prop) / 2.0
    else:
        dev_prop = float(dev_prop or 0.0)
        test_prop = float(test_prop or 0.0)

    if (dev_prop + test_prop) <= 0.0:
        dev_docs = []
        test_docs = rest_docs
    else:
        frac_dev = dev_prop / (dev_prop + test_prop)
        dev_docs, test_docs = stratified_split(rest_docs, frac_dev, seed + 1)

    # --- anti-fuite: éliminer les doublons de texte entre splits ---
    splits_clean, overlap_removed = _drop_overlap_by_text(
        {"train": train_docs_raw, "dev": dev_docs, "test": test_docs},
        order=["train", "dev", "test"],
    )
    train_docs_raw = splits_clean["train"]
    dev_docs = splits_clean["dev"]
    test_docs = splits_clean["test"]

    if sum(overlap_removed.values()) > 0:
        print(
            "[core_prepare] Anti-fuite (text-hash) -> doublons supprimés entre splits: "
            + ", ".join(f"{k}={v}" for k, v in overlap_removed.items())
        )

    # compat historique: job == test
    job_docs = test_docs

    label_counts_train_raw = Counter(d["label"] for d in train_docs_raw)
    label_counts_dev = Counter(d["label"] for d in dev_docs)
    label_counts_test = Counter(d["label"] for d in test_docs)

    # Appliquer l'équilibrage via la config V4 (balance.yml + params) uniquement sur train
    train_docs, label_counts_balanced = apply_balance(
        train_docs_raw,
        params,
        label_counts_train_raw,
    )

    balance_strategy = params.get("balance_strategy", "none")
    print(f"[core_prepare] Docs après équilibrage train ({balance_strategy}): {len(train_docs)}")
    print(f"[core_prepare] Répartition labels train (après balance): {label_counts_balanced}")

    print(f"[core_prepare] Split stratifié train/dev/test avec train_prop={train_prop}, dev_prop={dev_prop}, test_prop={test_prop}")
    print(f"  -> train (avant balance): {len(train_docs_raw)} docs | {label_counts_train_raw}")
    print(f"  -> dev  (naturel)      : {len(dev_docs)} docs | {label_counts_dev}")
    print(f"  -> test (naturel)      : {len(test_docs)} docs | {label_counts_test}")

    meta = {
        "profile": params.get("profile"),
        "dataset_id": dataset_id,
        "corpus_id": corpus_id,
        "view": view,
        "modality_filter": modality_filter,
        "label_field": label_field,
        "label_fields": label_fields,
        "label_map": label_map_path,
        "ideology_config": ideology_cfg,
        "actors_filter": actors_cfg,
        "source_field": source_field,
        "source_corpora": [c.get("corpus_id") for c in corpora_multi],
        "merge_mode": merge_mode,
        "balance_strategy": balance_strategy,
        "balance_preset": balance_preset,
        "train_prop": train_prop,
        "dev_prop": float(dev_prop),
        "test_prop": float(test_prop),
        "seed": seed,
        "tokenizer": tokenizer_name,
        "dedup_on": dedup_on,
        "ideology_mode": (ideology_cfg or {}).get("mode"),
        "ideology_view": (ideology_cfg or {}).get("view") or (ideology_cfg or {}).get("granularity"),
        "ideology_stats": dict(ideology_stats),
        "n_docs_raw": int(len(docs)),
        "n_docs_train_raw": int(len(train_docs_raw)),
        "n_docs_train_balanced": int(len(train_docs)),
        "n_docs_dev": int(len(dev_docs)),
        "n_docs_test": int(len(test_docs)),
        "n_docs_job": int(len(job_docs)),
        "overlap_removed": {k: int(v) for k, v in overlap_removed.items()},
        "label_counts_before": dict(label_counts),
        "label_counts_train_before_balance": dict(label_counts_train_raw),
        "label_counts_train_after": dict(label_counts_balanced),
        "label_counts_dev": dict(label_counts_dev),
        "label_counts_test": dict(label_counts_test),
        "label_counts_job": dict(label_counts_test),
        "modality_counts": dict(modality_counts),
        "params_hardware": params.get("hardware", {}),
        "pipeline_version": PIPELINE_VERSION,
    }
    if actor_counts_before:
        meta["actor_counts_before"] = dict(actor_counts_before)
    if actor_counts_after:
        meta["actor_counts_after"] = dict(actor_counts_after)
    if params.get("class_weights") is not None:
        meta["label_weights"] = params["class_weights"]

    if dry_run:
        print("[core_prepare] Dry-run activé, aucun TSV écrit.")
        return meta

    # Écriture TSV
    train_path = os.path.join(interim_dir, "train.tsv")
    dev_path = os.path.join(interim_dir, "dev.tsv")
    test_path = os.path.join(interim_dir, "test.tsv")
    job_path = os.path.join(interim_dir, "job.tsv")
    meta_path = os.path.join(interim_dir, "meta_view.json")

    write_tsv(train_path, train_docs, source_field=source_field)
    write_tsv(dev_path, dev_docs, source_field=source_field)
    write_tsv(test_path, test_docs, source_field=source_field)
    # compat : job.tsv = test.tsv
    write_tsv(job_path, job_docs, source_field=source_field)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[core_prepare] train.tsv écrit : {train_path}")
    print(f"[core_prepare] dev.tsv écrit   : {dev_path}")
    print(f"[core_prepare] test.tsv écrit  : {test_path}")
    print(f"[core_prepare] job.tsv écrit   : {job_path}")
    print(f"[core_prepare] meta_view.json  : {meta_path}")

    return meta


def write_tsv(path: str, docs: List[Dict[str, Any]], source_field: str = "corpus_id") -> None:
    """Écrire une liste de docs en TSV (id, label, label_raw, modality, source, text)."""
    fieldnames = ["id", "xml_id", "label", "label_raw", "modality", source_field, "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for d in docs:
            writer.writerow(
                {
                    "id": d["id"],
                    "xml_id": d.get("xml_id") or "",
                    "label": d["label"],
                    "label_raw": d["label_raw"],
                    "modality": d["modality"],
                    source_field: (d.get("meta") or {}).get(source_field, "unknown"),
                    "text": d["text"],
                }
            )



# Équilibrage

def rebalance_cap_docs(
    docs: List[Dict[str, Any]],
    cap: int,
    oversample: bool,
    offset: int,
) -> List[Dict[str, Any]]:
    """Reprise de la logique V2 pour cap_docs.

    - On regroupe par label.
    - Pour chaque label : on prend au plus `cap` docs (après rotation contrôlée).
    - Si oversample=True et qu'on a moins que cap, on duplique des docs jusqu'à cap.
    - On mélange à la fin.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    rng = random.Random(offset)
    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        n = len(L)
        if n == 0:
            continue
        start = (hash(lab) + offset) % n
        rot = L[start:] + L[:start]
        take = min(cap, n)
        out.extend(rot[:take])
        if oversample and n < cap:
            need = cap - n
            for i in range(need):
                dup = dict(rng.choice(L))
                dup["id"] = f'{dup["id"]}#dup{i+1}'
                out.append(dup)
    rng.shuffle(out)
    return out


def rebalance_cap_tokens(
    docs: List[Dict[str, Any]],
    cap_tokens: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Reprise V2 pour cap_tokens (par label).

    On sélectionne, par label, suffisamment de docs (et de duplications)
    pour atteindre ~cap_tokens tokens.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        n = len(L)
        if n == 0:
            continue
        start = (hash(lab) + offset) % n
        rot = L[start:] + L[:start]
        tot = 0
        buf: List[Dict[str, Any]] = []
        # Première passe : on prend tant qu'on n'a pas atteint cap_tokens
        for d in rot:
            if tot >= cap_tokens:
                break
            buf.append(d)
            tot += int(d.get("tokens") or 0)
        # Si on n'a toujours pas assez de tokens, on duplique
        i = 0
        while tot < cap_tokens and i < max(n, 1):
            dd = rot[i % n]
            dup = dict(dd)
            dup["id"] = f'{dd["id"]}#dup{i+1}'
            buf.append(dup)
            tot += int(dd.get("tokens") or 0)
            i += 1
        out.extend(buf)
    random.Random(offset).shuffle(out)
    return out


def rebalance_alpha_total(
    docs: List[Dict[str, Any]],
    alpha: float,
    total: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Reprise fidèle de rebalance_alpha_total de V2.

    - On calcule cnt(label).
    - Poids w(label) = (cnt(label) ** alpha).
    - Probabilité p(label) = w(label) / sum(w).
    - Quotas entiers par arrondi, ajustés pour total.
    - Échantillonnage avec duplication si besoin.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    # comptages
    cnt = {lab: len(L) for lab, L in by.items()}
    w = {lab: (c ** alpha) for lab, c in cnt.items() if c > 0}
    z = sum(w.values()) or 1.0
    probs = {lab: wv / z for lab, wv in w.items()}
    quotas = {lab: max(1, int(round(total * p))) for lab, p in probs.items()}

    # Ajuster quotas pour que la somme soit exactement total
    diff = total - sum(quotas.values())
    labs = list(quotas.keys())
    i = 0
    while diff != 0 and labs:
        quotas[labs[i % len(labs)]] += 1 if diff > 0 else -1
        i += 1
        diff = total - sum(quotas.values())

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        k = quotas.get(lab, 0)
        if k <= 0 or not L:
            continue
        if k <= len(L):
            out.extend(rng.sample(L, k))
        else:
            out.extend(L)
            for j in range(k - len(L)):
                dup = dict(rng.choice(L))
                dup["id"] = f'{dup["id"]}#dup{j+1}'
                out.append(dup)
    rng.shuffle(out)
    return out


def apply_balance(
    docs: List[Dict[str, Any]],
    params: Dict[str, Any],
    label_counts: Counter,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Stratégies : none | cap_docs | cap_tokens | alpha_total | class_weights
    """
    strategy = (params.get("balance_strategy") or "none").strip().lower()
    balance_cfg = params.get("balance_cfg", {}) or {}
    seed = int(params.get("seed_resolved") or (parse_seed(params.get("seed"), default=42) or 42))

    # Chercher le preset de façon robuste :
    # 1) balance_cfg["strategies"][strategy]["presets"][balance_preset]
    # 2) sinon balance_cfg["presets"][balance_preset]
    preset_name = params.get("balance_preset")
    preset = {}
    try:
        preset = balance_cfg.get("strategies", {}).get(strategy, {}).get("presets", {}).get(preset_name, {})
    except Exception:
        preset = {}
    if not preset and preset_name:
        preset = balance_cfg.get("presets", {}).get(preset_name, {})  # fallback

    if strategy in ("none", "", None):
        return docs, label_counts

    if strategy == "cap_docs":
        cap = int(preset.get("cap_per_label") or preset.get("cap") or 0)
        oversample = bool(preset.get("oversample", False))
        offset = int(preset.get("offset", 0))
        if cap <= 0:
            return docs, label_counts
        new_docs = rebalance_cap_docs(docs=docs, cap=cap, oversample=oversample, offset=offset)
        return new_docs, Counter(d["label"] for d in new_docs)

    if strategy == "cap_tokens":
        cap_tokens = int(preset.get("cap_tokens_per_label") or preset.get("cap_tokens") or 0)
        offset = int(preset.get("offset", 0))
        if cap_tokens <= 0:
            return docs, label_counts
        new_docs = rebalance_cap_tokens(docs=docs, cap_tokens=cap_tokens, offset=offset)
        return new_docs, Counter(d["label"] for d in new_docs)

    if strategy == "alpha_total":
        alpha = float(preset.get("alpha", 1.0))
        total_docs = int(preset.get("total_docs", len(docs)))
        new_docs = rebalance_alpha_total(docs=docs, alpha=alpha, total=total_docs, seed=seed)
        return new_docs, Counter(d["label"] for d in new_docs)

    if strategy == "class_weights":
        params["class_weights"] = compute_class_weights_from_counts(label_counts)
        return docs, label_counts

    return docs, label_counts





# Formats (TSV -> formats modèles)


def build_formats(params: Dict[str, Any], meta_view: Dict[str, Any]) -> None:
    """
    Construit les formats pour chaque famille:
      - spaCy : DocBin shardés (train_000.spacy, ...)
      - sklearn / hf / check : référencent les TSV.
    Ecrit data/processed/<dataset>/<view>/meta_formats.json
    """
    from scripts.core.core_utils import log

    families = params.get("families", []) or []
    corpus_id = params.get("corpus_id", params.get("corpus", {}).get("corpus_id", "unknown_corpus"))
    dataset_id = params.get("dataset_id") or corpus_id
    view = params.get("view", "unknown_view")

    interim_dir = Path("data") / "interim" / str(dataset_id) / view
    processed_root = Path("data") / "processed" / str(dataset_id) / view
    processed_root.mkdir(parents=True, exist_ok=True)

    train_tsv = interim_dir / "train.tsv"
    dev_tsv   = interim_dir / "dev.tsv"
    test_tsv  = interim_dir / "test.tsv"
    job_tsv   = interim_dir / "job.tsv"  # compat (alias test)

    formats_meta = {
        "profile": params.get("profile"),
        "dataset_id": dataset_id,
        "corpus_id": corpus_id,
        "view": view,
        "families": {},
    }

    # spaCy (DocBin) : optionnel
    # - Sur gros corpus, on veut pouvoir faire un prepare TSV-first sans spaCy.
    # - Si spaCy n'est pas installé ou si generate_spacy_docbin=false, on log et on continue.
    generate_spacy_docbin = bool(params.get("generate_spacy_docbin", True))
    if "spacy" in families and not generate_spacy_docbin:
        log("prepare", "spacy", "SKIP DocBin (generate_spacy_docbin=false) → TSV OK, formats spaCy non générés")
        formats_meta["families"]["spacy"] = {
            "enabled": False,
            "reason": "generate_spacy_docbin=false",
            "train_tsv": str(train_tsv),
            "dev_tsv": str(dev_tsv),
            "test_tsv": str(test_tsv),
            "job_tsv": str(job_tsv),
        }

    if "spacy" in families and generate_spacy_docbin:
        # import paresseux
        try:
            import spacy
            from spacy.tokens import DocBin
        except Exception as e:
            log(
                "prepare",
                "spacy",
                f"WARNING: spaCy indisponible → DocBin non générés (TSV OK). Détail: {e}",
            )
            formats_meta["families"]["spacy"] = {
                "enabled": False,
                "reason": f"spacy_import_error: {e}",
                "train_tsv": str(train_tsv),
                "dev_tsv": str(dev_tsv),
                "test_tsv": str(test_tsv),
                "job_tsv": str(job_tsv),
            }
        else:
            models_cfg = params.get("models_cfg", {}).get("families", {}).get("spacy", {})
            models_spacy = params.get("models_spacy") or []

            # Déterminer la langue (premier modèle qui en fournit une)
            lang = "fr"
            for mid in models_spacy:
                mcfg = models_cfg.get(mid) or {}
                if "lang" in mcfg:
                    lang = mcfg["lang"]
                    break

            spacy_dir = processed_root / "spacy"
            spacy_dir.mkdir(parents=True, exist_ok=True)

            shard_docs = int(params.get("hardware", {}).get("spacy_shard_docs", 0)) or 0
            if shard_docs < 1:
                shard_docs = 0  # pas de sharding

            # Gestion des très longs textes (outliers WEB) : protège la génération DocBin.
            # Remarque: cela n'affecte que la famille spaCy (pas les TSV sklearn/check/hf).
            spacy_max_chars = int(
                params.get("spacy_max_chars", 0)
                or (params.get("hardware", {}) or {}).get("spacy_max_chars", 0)
                or 1_000_000
            )
            spacy_long_text_policy = str(params.get("spacy_long_text_policy", "drop") or "drop").lower()
            # Politique sur les très longs textes :
            # - drop     : on ignore le doc (safe, recommandé en low-spec)
            # - truncate : on coupe à spacy_max_chars
            # - keep     : on tente de garder (et on ajuste nlp.max_length au besoin)
            # - raise    : on stoppe net (mode "strict")
            if spacy_long_text_policy not in ("drop", "truncate", "keep", "raise"):
                spacy_long_text_policy = "drop"
            spacy_long_stats = {"dropped": 0, "truncated": 0, "max_seen": 0, "errors": 0}

            def _collect_labels_from_tsv(tsv_path: Path) -> set:
                labels = set()
                if not tsv_path.exists():
                    return labels
                with tsv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        lab = (row.get("label") or "").strip()
                        if lab:
                            labels.add(lab)
                return labels

            def _build_docbins(tsv_path: Path, prefix: str, labels_all: list) -> Tuple[List[str], int]:
                """Retourne: (docbin_paths, total_docs)."""
                nlp = spacy.blank(lang)
                # Alignement avec spaCy: empêcher E088 sur textes > nlp.max_length.
                try:
                    if spacy_max_chars and spacy_max_chars > 0:
                        nlp.max_length = max(int(getattr(nlp, 'max_length', 0) or 0), int(spacy_max_chars) + 1)
                except Exception:
                    pass
                paths: List[str] = []
                total = 0
                shard_idx = 0
                db = DocBin(store_user_data=True)
                docs_in_shard = 0

                with tsv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        text = row.get("text") or ""
                        label = row.get("label")
                        if not text or not label:
                            continue
                        n_chars = len(text)
                        if n_chars > spacy_long_stats["max_seen"]:
                            spacy_long_stats["max_seen"] = n_chars
                        if spacy_max_chars and n_chars > spacy_max_chars:
                            if spacy_long_text_policy == "drop":
                                spacy_long_stats["dropped"] += 1
                                continue
                            elif spacy_long_text_policy == "truncate":
                                spacy_long_stats["truncated"] += 1
                                text = text[:spacy_max_chars]
                            elif spacy_long_text_policy == "keep":
                                # On tente de garder le texte (utile si spacy_max_chars est un "soft cap").
                                pass
                            else:
                                raise ValueError(
                                    f"[prepare:spacy] texte trop long: len={n_chars} > spacy_max_chars={spacy_max_chars} (policy=raise)"
                                )

                        # Assure une compatibilité totale avec des documents très longs.
                        # Si ça devient trop lourd pour la machine, l'exception est capturée plus bas et le doc est ignoré.
                        if hasattr(nlp, "max_length") and len(text) >= int(getattr(nlp, "max_length", 0) or 0):
                            nlp.max_length = len(text) + 1

                        try:
                            doc = nlp.make_doc(text)
                        except Exception:
                            spacy_long_stats["errors"] += 1
                            continue
                        doc.cats = {lab: (1.0 if lab == label else 0.0) for lab in labels_all}
                        db.add(doc)
                        total += 1
                        docs_in_shard += 1

                        # flush shard si besoin
                        if shard_docs and docs_in_shard >= shard_docs:
                            outp = spacy_dir / f"{prefix}_{shard_idx:03d}.spacy"
                            db.to_disk(outp)
                            paths.append(str(outp))
                            shard_idx += 1
                            db = DocBin(store_user_data=True)
                            docs_in_shard = 0

                # flush dernier shard
                if docs_in_shard > 0 or (not shard_docs and total > 0):
                    outp = spacy_dir / (
                        f"{prefix}_{shard_idx:03d}.spacy" if shard_docs else f"{prefix}.spacy"
                    )
                    db.to_disk(outp)
                    paths.append(str(outp))

                return paths, total

            log("prepare", "spacy", "Construction des DocBin...")
            # union des labels sur train/dev/test pour garantir des keys stables
            labels_all = sorted(
                _collect_labels_from_tsv(train_tsv)
                | _collect_labels_from_tsv(dev_tsv if dev_tsv.exists() else job_tsv)
                | _collect_labels_from_tsv(test_tsv if test_tsv.exists() else job_tsv)
            )
            train_paths, n_train = _build_docbins(train_tsv, "train", labels_all)
            dev_paths, n_dev = _build_docbins(dev_tsv if dev_tsv.exists() else job_tsv, "dev", labels_all)
            test_paths, n_test = _build_docbins(test_tsv if test_tsv.exists() else job_tsv, "test", labels_all)
            # compat: job = test
            job_paths = list(test_paths)
            n_job = n_test

            formats_meta["families"]["spacy"] = {
                "enabled": True,
                "dir": str(spacy_dir),
                "train_spacy": train_paths,
                "dev_spacy": dev_paths,
                "test_spacy": test_paths,
                "job_spacy": job_paths,
                "labels_set": labels_all,
                "lang": lang,
                "n_train_docs": n_train,
                "n_dev_docs": n_dev,
                "n_test_docs": n_test,
                "n_job_docs": n_job,
                "spacy_shard_docs": shard_docs,
                "spacy_max_chars": spacy_max_chars,
                "spacy_long_text_policy": spacy_long_text_policy,
                "spacy_long_text_stats": spacy_long_stats,
            }

    # sklearn / hf / check : référence TSV
    for fam in ("sklearn", "hf", "check"):
        if fam in families:
            formats_meta["families"][fam] = {
                "train_tsv": str(train_tsv),
                "dev_tsv": str(dev_tsv),
                "test_tsv": str(test_tsv),
                "job_tsv": str(job_tsv),
            }

    meta_path = processed_root / "meta_formats.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(formats_meta, f, ensure_ascii=False, indent=2)

    log("prepare", "formats", f"Formats écrits → {meta_path}")








# main

def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    from scripts.core.core_utils import apply_global_seed, log
    seed_applied = apply_global_seed(params.get("seed"))
    log("prepare", "seed", f"Global seed: {'appliquée' if seed_applied else 'non appliquée'} ({params.get('seed')})")
    log("prepare", "info", f"tokenizer={params.get('tokenizer','split')} dedup_on={params.get('dedup_on','none')}")

    meta_view = build_view(params, dry_run=args.dry_run)
    if not args.dry_run:
        build_formats(params, meta_view)


if __name__ == "__main__":
    main()



