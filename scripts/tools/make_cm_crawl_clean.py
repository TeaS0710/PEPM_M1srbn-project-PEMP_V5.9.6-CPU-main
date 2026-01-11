import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

TEST_TSV   = "data/interim/FINAL_web1_crawl/crawl/test.tsv"
MODEL_PATH = "models/FINAL_web1_crawl/crawl/sklearn/tfidf_svm_quick/model.joblib"

OUT_PNG = "reports/figures/cm_FINAL_web1_crawl_crawl_tfidf_svm_quick_norm.png"
OUT_MAP = "reports/figures/cm_FINAL_web1_crawl_crawl_tfidf_svm_quick_labels_map.txt"
OUT_TOP = "reports/figures/cm_FINAL_web1_crawl_crawl_tfidf_svm_quick_top_confusions.tsv"

CHUNK = 2000  # prédiction par paquets (évite pics RAM)

def find(obj, pred):
    if pred(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = find(v, pred)
            if r is not None:
                return r
    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = find(v, pred)
            if r is not None:
                return r
    return None

def short(lbl: str) -> str:
    s = lbl.replace("crawl_", "").replace("asr_party_", "")
    s = re.sub(r"_[0-9]{8}_[0-9]{6}$", "", s)  # suffix dates
    s = s.replace("_full", "")
    s = s.replace("initiative_communiste", "init_com")
    return s[:18]

df = pd.read_csv(TEST_TSV, sep="\t")
y_true = df["label"].astype(str).to_numpy()
texts  = df["text"].astype(str).tolist()

payload = joblib.load(MODEL_PATH)

# 1) pipeline complète si disponible
pipe = find(payload, lambda x: isinstance(x, Pipeline) or (hasattr(x, "steps") and hasattr(x, "predict")))

# 2) fallback : vectorizer + classifier
vec = find(payload, lambda x: hasattr(x, "transform") and hasattr(x, "vocabulary_"))
clf = find(payload, lambda x: hasattr(x, "predict") and not hasattr(x, "transform") and not hasattr(x, "steps"))

if pipe is None and (vec is None or clf is None):
    keys = list(payload.keys()) if isinstance(payload, dict) else None
    raise RuntimeError(f"Impossible de retrouver pipeline ou (vectorizer+clf) dans {MODEL_PATH}. type={type(payload)} keys={keys}")

# prédiction chunkée
preds = []
for i in range(0, len(texts), CHUNK):
    chunk = texts[i:i+CHUNK]
    if pipe is not None:
        y_pred_chunk = pipe.predict(chunk)
    else:
        X = vec.transform(chunk)
        y_pred_chunk = clf.predict(X)
    preds.append(np.asarray(y_pred_chunk, dtype=object))
y_pred = np.concatenate(preds, axis=0)

labels = sorted(set(y_true) | set(y_pred))
short_map = {l: short(l) for l in labels}

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

# mapping labels courts
with open(OUT_MAP, "w", encoding="utf-8") as f:
    for l in labels:
        f.write(f"{short_map[l]}\t{l}\n")

# top confusions (hors diagonale)
pairs = []
for i, lt in enumerate(labels):
    row_sum = cm[i].sum()
    for j, lp in enumerate(labels):
        if i == j:
            continue
        c = int(cm[i, j])
        if c > 0:
            pairs.append((c, (c / row_sum) if row_sum else 0.0, lt, lp))
pairs.sort(reverse=True, key=lambda x: x[0])

with open(OUT_TOP, "w", encoding="utf-8") as f:
    f.write("count\trow_share\ttrue\tpred\n")
    for c, r, lt, lp in pairs[:40]:
        f.write(f"{c}\t{r:.4f}\t{lt}\t{lp}\n")

# plot lisible
plt.figure(figsize=(14, 14), dpi=220)
ax = plt.gca()
im = ax.imshow(cm_norm, aspect="auto")
ax.set_title("Confusion matrix (row-normalized) — crawl (20 classes)", pad=16)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

tick = np.arange(len(labels))
ax.set_xticks(tick)
ax.set_yticks(tick)
ax.set_xticklabels([short_map[l] for l in labels], rotation=60, ha="right", fontsize=8)
ax.set_yticklabels([short_map[l] for l in labels], fontsize=8)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row share", rotation=90)

# n’affiche que les gros pourcentages
thr = 0.20
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        v = cm_norm[i, j]
        if v >= thr:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")

print("OK ->", OUT_PNG)
print("label map ->", OUT_MAP)
print("top confusions ->", OUT_TOP)
