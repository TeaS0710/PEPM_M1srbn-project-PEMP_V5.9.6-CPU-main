import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

TEST_TSV   = "data/interim/FINAL_mix_ideo/ideology_global/test.tsv"
MODEL_PATH = "models/FINAL_mix_ideo/ideology_global/sklearn/tfidf_svm_quick/model.joblib"
OUT_PNG    = "reports/figures/scatter_FINAL_mix_ideo_ideology_global_tfidf_svm_quick_svd2_clean.png"

SEED = 52
MAX_WEB = 6000

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

df = pd.read_csv(TEST_TSV, sep="\t")
df["label"] = df["label"].astype(str)
df["corpus_id"] = df["corpus_id"].astype(str)

# subsample AVANT vectorisation (sinon inutilement lourd)
is_asr = df["corpus_id"].str.lower().eq("asr1")
asr_df = df[is_asr]
web_df = df[~is_asr]
if len(web_df) > MAX_WEB:
    web_df = web_df.sample(n=MAX_WEB, random_state=SEED)
plot_df = pd.concat([asr_df, web_df], axis=0).reset_index(drop=True)

payload = joblib.load(MODEL_PATH)

# si pipeline dispo : récupérer le vectorizer dedans
pipe = find(payload, lambda x: isinstance(x, Pipeline) or (hasattr(x, "steps") and hasattr(x, "predict")))
vec = find(payload, lambda x: hasattr(x, "transform") and hasattr(x, "vocabulary_"))

if vec is None and pipe is not None and hasattr(pipe, "steps"):
    for _, step in pipe.steps:
        if hasattr(step, "transform") and hasattr(step, "vocabulary_"):
            vec = step
            break

if vec is None:
    keys = list(payload.keys()) if isinstance(payload, dict) else None
    raise RuntimeError(f"Impossible de retrouver un TF-IDF vectorizer dans {MODEL_PATH}. type={type(payload)} keys={keys}")

X = vec.transform(plot_df["text"].astype(str).tolist())
Z = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(X)
plot_df["svd1"], plot_df["svd2"] = Z[:,0], Z[:,1]

# axes robustes (évite outliers)
x1, x2 = np.percentile(plot_df["svd1"], [1, 99])
y1, y2 = np.percentile(plot_df["svd2"], [1, 99])

plt.figure(figsize=(12, 8), dpi=220)
ax = plt.gca()
ax.set_title("TF-IDF + TruncatedSVD(2) — ideology_global (test) — mix web+ASR", pad=12)
ax.set_xlabel("SVD-1")
ax.set_ylabel("SVD-2")

for (lab, corp), g in plot_df.groupby(["label", "corpus_id"]):
    marker = "o" if corp.lower() == "web1" else "^"
    ax.scatter(g["svd1"], g["svd2"], s=10, alpha=0.35, marker=marker, label=f"{lab}/{corp}")

ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.legend(loc="best", fontsize=8, frameon=True)
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
print("OK ->", OUT_PNG)
