#!/usr/bin/env python3
import argparse, csv, hashlib, json, os, re, sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# Gros champs CSV possibles -> éviter "field larger than field limit".
try:
    csv.field_size_limit(2**31 - 1)
except Exception:
    pass

NS_TEI = "http://www.tei-c.org/ns/1.0"
NS_XML = "http://www.w3.org/XML/1998/namespace"
ns = {"tei": NS_TEI}

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

def norm_text(s: str) -> str:
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    return s

def norm_text_strong(s: str) -> str:
    s = norm_text(s)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def itertext_excluding(elem, exclude_pred):
    # Recursive itertext with subtree exclusion (ElementTree has no parent pointers)
    if exclude_pred(elem):
        return
    if elem.text:
        yield elem.text
    for child in list(elem):
        yield from itertext_excluding(child, exclude_pred)
        if child.tail:
            yield child.tail

def tei_iter(path):
    # Stream over TEI nodes to handle large corpora
    tei_tag = f"{{{NS_TEI}}}TEI"
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == tei_tag:
            yield el
            el.clear()

def get_xml_id(tei):
    return tei.get(f"{{{NS_XML}}}id") or tei.get("xml:id") or ""

def find_terms(tei, ttype):
    out = []
    for t in tei.findall(f".//tei:term[@type='{ttype}']", ns):
        if t.text:
            v = t.text.strip()
            if v:
                out.append(v)
    return out

def find_classcodes(tei):
    out = []
    for cc in tei.findall(".//tei:classCode", ns):
        scheme = (cc.get("scheme") or "").strip()
        val = (cc.text or "").strip()
        if scheme or val:
            out.append((scheme, val))
    return out

def extract_text_all(tei):
    text_el = tei.find(".//tei:text", ns)
    if text_el is None:
        return ""
    return " ".join(text_el.itertext()).strip()

def extract_text_no_segments(tei):
    text_el = tei.find(".//tei:text", ns)
    if text_el is None:
        return ""
    ab_tag = f"{{{NS_TEI}}}ab"
    def exclude_pred(e):
        return (e.tag == ab_tag) and (e.get("type") == "segments")
    return " ".join(itertext_excluding(text_el, exclude_pred)).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tei_path", help="Path to TEI corpus.xml")
    ap.add_argument("--name", default="corpus", help="Corpus name for reporting")
    ap.add_argument("--out_json", default=None, help="Write summary JSON here")
    ap.add_argument("--out_csv_issues", default=None, help="Write per-doc issues CSV here")
    ap.add_argument("--fail_on_critical", action="store_true", help="Exit 2 if critical issues found")
    args = ap.parse_args()

    # Accumulators
    n_docs = 0
    missing_id = 0
    dup_id = 0
    empty_text = 0
    missing_text_node = 0

    ids_seen = set()
    id_dups = []

    hash_seen = {}          # strong normalized hash -> first xml:id
    dup_text_exact = 0
    dup_pairs = []

    schemes = Counter()
    classcode_vals = Counter()
    domains = Counter()
    crawls = Counter()
    modalities = Counter()
    domain_to_labels = defaultdict(set)

    # leakage-ish counters
    domain_in_text = 0
    crawl_in_text = 0
    classcode_in_text = 0
    segments_present = 0
    segments_dup_risky = 0

    issues_rows = []

    try:
        for tei in tei_iter(args.tei_path):
            n_docs += 1
            xid = get_xml_id(tei).strip()
            if not xid:
                missing_id += 1
                xid = f"__missing_id__:{n_docs}"

            if xid in ids_seen:
                dup_id += 1
                id_dups.append(xid)
            ids_seen.add(xid)

            # metadata used by pipeline
            doms = find_terms(tei, "domain")
            crws = find_terms(tei, "crawl")
            mods = find_terms(tei, "modality")
            for d in doms: domains[d] += 1
            for c in crws: crawls[c] += 1
            for m in mods: modalities[m] += 1

            ccs = find_classcodes(tei)
            for scheme, val in ccs:
                schemes[scheme] += 1
                classcode_vals[(scheme, val)] += 1

            # text extraction variants (to detect ASR double-text)
            txt_all = extract_text_all(tei)
            txt_noseg = extract_text_no_segments(tei)

            if not txt_all and not txt_noseg:
                # either no <text> or empty
                if tei.find(".//tei:text", ns) is None:
                    missing_text_node += 1
                empty_text += 1

            # detect segments duplication risk
            has_segments = tei.find(".//tei:ab[@type='segments']", ns) is not None
            if has_segments:
                segments_present += 1
                la = len(txt_all)
                ln = len(txt_noseg)
                # If both exist and all_text much larger => likely duplication
                if ln > 0 and la / max(1, ln) > 1.25:
                    segments_dup_risky += 1

            # duplicate exact text (strong normalization)
            norm = norm_text_strong(txt_all)
            h = sha1(norm)
            if norm:
                if h in hash_seen:
                    dup_text_exact += 1
                    if len(dup_pairs) < 200:
                        dup_pairs.append((xid, hash_seen[h]))
                else:
                    hash_seen[h] = xid

            # leakage heuristics: metadata showing up in text
            tl = norm_text(txt_all)
            leak_dom = any(d.lower() in tl for d in doms if len(d) >= 4)
            leak_crw = any(c.lower() in tl for c in crws if len(c) >= 4)
            leak_cc  = any((val.lower() in tl) for (scheme, val) in ccs if val and len(val) >= 4)

            if leak_dom: domain_in_text += 1
            if leak_crw: crawl_in_text += 1
            if leak_cc:  classcode_in_text += 1

            # domain->labels structure (web leakage-by-design detector)
            # Choose one "primary label" heuristically: first non-empty classCode value
            label_vals = [val for (_, val) in ccs if val]
            for d in doms:
                for lv in label_vals[:1]:
                    domain_to_labels[d].add(lv)

            # record per-doc issues
            doc_issues = []
            if xid.startswith("__missing_id__"):
                doc_issues.append("missing_xml_id")
            if leak_dom:
                doc_issues.append("domain_in_text")
            if leak_crw:
                doc_issues.append("crawl_in_text")
            if leak_cc:
                doc_issues.append("classcode_in_text")
            if has_segments and (len(txt_noseg) > 0) and (len(txt_all) / max(1, len(txt_noseg)) > 1.25):
                doc_issues.append("segments_dup_risky")
            if not txt_all.strip():
                doc_issues.append("empty_text_all")

            if doc_issues:
                issues_rows.append({
                    "xml_id": xid,
                    "issues": ";".join(doc_issues),
                    "len_all": len(txt_all),
                    "len_no_segments": len(txt_noseg),
                    "n_domains": len(doms),
                    "n_crawls": len(crws),
                    "n_classcodes": len(ccs),
                })

    except ET.ParseError as e:
        print(f"[CRITICAL] XML parse error: {e}", file=sys.stderr)
        sys.exit(2)

    # domain purity
    domain_pure = sum(1 for d, labs in domain_to_labels.items() if len(labs) <= 1)
    domain_total = len(domain_to_labels)
    domain_pure_ratio = (domain_pure / domain_total) if domain_total else 0.0

    summary = {
        "corpus": args.name,
        "tei_path": args.tei_path,
        "n_docs": n_docs,
        "missing_xml_id": missing_id,
        "duplicate_xml_id": dup_id,
        "missing_text_node": missing_text_node,
        "empty_text_docs": empty_text,
        "segments_present_docs": segments_present,
        "segments_dup_risky_docs": segments_dup_risky,
        "duplicate_text_exact_docs": dup_text_exact,
        "leak_domain_in_text_docs": domain_in_text,
        "leak_crawl_in_text_docs": crawl_in_text,
        "leak_classcode_in_text_docs": classcode_in_text,
        "top_domains": domains.most_common(10),
        "top_crawls": crawls.most_common(10),
        "top_modalities": modalities.most_common(10),
        "top_classcodes": [(k[0], k[1], v) for k, v in classcode_vals.most_common(10)],
        "domain_label_purity": {
            "domains_total": domain_total,
            "domains_pure_or_single_label": domain_pure,
            "purity_ratio": domain_pure_ratio
        },
        "sample_duplicate_pairs": dup_pairs[:20],
        "sample_duplicate_xml_ids": id_dups[:20],
        "issues_count": len(issues_rows),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Créer les répertoires de sortie si besoin
    for outp in (args.out_json, args.out_csv_issues):
        if outp:
            d = os.path.dirname(outp)
            if d:
                os.makedirs(d, exist_ok=True)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.out_csv_issues:
        with open(args.out_csv_issues, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(issues_rows[0].keys()) if issues_rows else
                               ["xml_id","issues","len_all","len_no_segments","n_domains","n_crawls","n_classcodes"])
            w.writeheader()
            for r in issues_rows:
                w.writerow(r)

    # Decide criticality
    critical = []
    if dup_id > 0: critical.append("duplicate_xml_id")
    if dup_text_exact > 0: critical.append("duplicate_text_exact")
    # segments duplication is critical mainly for ASR because it biases length & signal
    if segments_dup_risky > 0: critical.append("segments_dup_risky")
    # domain purity high is not a "bug" but a major bias
    if domain_total and domain_pure_ratio > 0.95: critical.append("domain_label_purity_high")

    if critical:
        print(f"\n[ALERT] Flags: {', '.join(critical)}", file=sys.stderr)
        if args.fail_on_critical:
            sys.exit(2)

if __name__ == "__main__":
    main()
