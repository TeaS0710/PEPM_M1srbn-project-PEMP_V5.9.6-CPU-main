#Projet PEPM By Yi Fan && Adrien
#!/usr/bin/env python3
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

def norm_text(t: str) -> str:
    t = (t or "").replace("\xa0", " ").replace("\t _blank", " ").replace("\u200b", "").strip()
    t = " ".join(t.split())
    return t

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()

def read_tsv(tsv: Path) -> Iterable[Tuple[str, str]]:
    if not tsv.exists():
        print(f"[ERR] TSV manquant: {tsv}", file=sys.stderr)
        return []
    with tsv.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        _ = next(rdr, None)  # header
        for row in rdr:
            if not row:
                continue
            txt = norm_text(row[0]) if len(row) > 0 else ""
            lab = (row[1].strip() if len(row) > 1 else "")
            if not txt or not lab:
                continue
            yield txt, lab

def percentiles(vals: List[int], ps=(50, 90)) -> Dict[str, float]:
    if not vals:
        return {str(p): 0.0 for p in ps}
    vals = sorted(vals)
    out: Dict[str, float] = {}
    for p in ps:
        k = (p / 100) * (len(vals) - 1)
        i = math.floor(k)
        j = math.ceil(k)
        if i == j:
            out[str(p)] = float(vals[i])
        else:
            out[str(p)] = float(vals[i] + (vals[j] - vals[i]) * (k - i))
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", nargs="+", required=True, help="Un ou plusieurs TSV (col0=text, col1=label)")
    ap.add_argument("--out-prefix", type=Path, default=Path("reports/corpus"))
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    label_counts_total = Counter()
    dup_by_label = Counter()
    lengths_by_label = defaultdict(list)
    seen_hash = set()
    doc_total = 0

    any_file = False
    for tsv in args.tsv:
        p = Path(tsv)
        if not p.exists():
            print(f"[WARN] Ignoré (introuvable): {p}", file=sys.stderr)
            continue
        any_file = True
        for txt, lab in read_tsv(p):
            doc_total += 1
            label_counts_total[lab] += 1
            lengths_by_label[lab].append(len(txt.split()))
            h = md5(txt)
            if h in seen_hash:
                dup_by_label[lab] += 1
            else:
                seen_hash.add(h)

    if not any_file:
        print("[ERR] Aucun TSV valide fourni.", file=sys.stderr)
        sys.exit(1)

    # label_dist.csv
    out_csv = args.out_prefix.with_suffix(".label_dist.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "count"])
        for lab, c in label_counts_total.most_common():
            w.writerow([lab, c])

    # stats json
    stats = {
        "docs": int(doc_total),
        "labels": int(len(label_counts_total)),
        "dups_total": int(sum(dup_by_label.values())),
        "dups_per_label": {k: int(v) for k, v in dup_by_label.items()},
        "length_tokens": {
            lab: {
                "n": len(L),
                "median": percentiles(L, [50])["50"] if L else 0.0,
                "p90": percentiles(L, [90])["90"] if L else 0.0,
            }
            for lab, L in lengths_by_label.items()
        },
    }
    out_json = args.out_prefix.with_suffix(".stats.json")
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] stats → {out_json}")
    print(f"[OK] label dist → {out_csv}")

if __name__ == "__main__":
    main()
