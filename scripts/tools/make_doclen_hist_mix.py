import pandas as pd
import matplotlib.pyplot as plt

TEST_TSV="data/interim/FINAL_mix_ideo/ideology_global/test.tsv"
OUT_PNG="reports/figures/doclen_FINAL_mix_ideo_ideology_global_by_corpus.png"

df = pd.read_csv(TEST_TSV, sep="\t")
df["corpus_id"] = df["corpus_id"].astype(str)
df["n_words"] = df["text"].astype(str).str.split().str.len()

plt.figure(figsize=(10,6), dpi=220)
ax = plt.gca()
for corp, g in df.groupby("corpus_id"):
    ax.hist(g["n_words"], bins=60, alpha=0.5, label=corp)

ax.set_title("Document length distribution (test) — words (whitespace)")
ax.set_xlabel("Number of words")
ax.set_ylabel("Count")
ax.set_xlim(0, df["n_words"].quantile(0.99))
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
print("OK ->", OUT_PNG)
