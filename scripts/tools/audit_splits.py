#!/usr/bin/env python3
import argparse, csv, glob, hashlib, json, os, re, sys
from collections import defaultdict

# Gros champs TSV possibles (text) -> éviter l'exception "field larger than field limit".
try:
    csv.field_size_limit(2**31 - 1)
except Exception:
    pass

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

def norm_text_strong(s: str) -> str:
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing TSV splits (train/dev/test/job).tsv")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--fail_on_leak", action="store_true")
    # Backward-compatible alias (certains Makefiles utilisent --strict)
    ap.add_argument("--strict", action="store_true", help="Alias de --fail_on_leak")
    args = ap.parse_args()

    if args.strict:
        args.fail_on_leak = True

    split_files = {}
    for name in ("train","dev","test","job"):
        p = os.path.join(args.dir, f"{name}.tsv")
        if os.path.exists(p):
            split_files[name] = p

    if not split_files:
        print(f"[CRITICAL] No split TSV found in {args.dir}", file=sys.stderr)
        sys.exit(2)

    # read each split
    by_split_ids = defaultdict(set)
    by_split_hash = defaultdict(set)
    by_split_rows = defaultdict(int)

    col_text = None
    col_id = None

    def pick_columns(fieldnames):
        nonlocal col_text, col_id
        fn = [f.strip() for f in fieldnames]
        # common conventions
        for cand in ("text","doc","content"):
            if cand in fn:
                col_text = cand
                break
        if col_text is None:
            # fallback: longest-named or last column often is text
            col_text = fn[-1]
        for cand in ("id","xml_id","doc_id"):
            if cand in fn:
                col_id = cand
                break

    for split, path in split_files.items():
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            if r.fieldnames is None:
                print(f"[CRITICAL] {path} has no header", file=sys.stderr)
                sys.exit(2)
            if col_text is None:
                pick_columns(r.fieldnames)

            for row in r:
                by_split_rows[split] += 1
                rid = (row.get(col_id) or f"__no_id__:{by_split_rows[split]}").strip()
                txt = (row.get(col_text) or "").strip()

                by_split_ids[split].add(rid)

                h = sha1(norm_text_strong(txt))
                if txt:
                    by_split_hash[split].add(h)

    # compute overlaps
    splits = list(split_files.keys())

    # Dans ce projet, job.tsv est souvent un alias de test.tsv (compat).
    # Dans ce cas, on n'interprète pas les overlaps test<->job comme une fuite.
    job_is_test_alias = False
    splits_for_overlap = splits
    if 'job' in splits and 'test' in splits:
        if by_split_ids['job'] == by_split_ids['test'] and by_split_hash['job'] == by_split_hash['test']:
            job_is_test_alias = True
            splits_for_overlap = [s for s in splits if s != 'job']

    overlap = []
    leak = False
    for i in range(len(splits_for_overlap)):
        for j in range(i+1, len(splits_for_overlap)):
            a, b = splits_for_overlap[i], splits_for_overlap[j]
            id_inter = by_split_ids[a] & by_split_ids[b]
            h_inter = by_split_hash[a] & by_split_hash[b]
            if id_inter or h_inter:
                leak = True
            overlap.append({
                "pair": f"{a}__{b}",
                "id_overlap": len(id_inter),
                "text_hash_overlap": len(h_inter),
                "sample_ids": list(sorted(list(id_inter))[:10]),
            })

    summary = {
        "dir": args.dir,
        "detected_columns": {"id": col_id, "text": col_text},
        "splits": {s: {"rows": by_split_rows[s], "unique_ids": len(by_split_ids[s]), "unique_text_hash": len(by_split_hash[s]), **({"alias_of": "test"} if (job_is_test_alias and s=="job") else {})}
                   for s in splits},
        "job_is_test_alias": job_is_test_alias,
        "overlap": overlap,
        "leak_detected": leak,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if leak:
        print("\n[ALERT] Overlap detected between splits (ID or text hash).", file=sys.stderr)
        if args.fail_on_leak:
            sys.exit(2)

if __name__ == "__main__":
    main()
