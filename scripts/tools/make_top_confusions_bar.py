import pandas as pd
import matplotlib.pyplot as plt

IN_TSV="reports/figures/cm_FINAL_web1_crawl_crawl_tfidf_svm_quick_top_confusions.tsv"
OUT_PNG="reports/figures/top_confusions_FINAL_web1_crawl.png"

df = pd.read_csv(IN_TSV, sep="\t").head(15)
df["pair"] = df["true"].astype(str) + " → " + df["pred"].astype(str)

plt.figure(figsize=(12,6), dpi=220)
ax = plt.gca()
ax.barh(df["pair"][::-1], df["row_share"][::-1])
ax.set_title("Top confusions (row share) — crawl")
ax.set_xlabel("Share of true-class predicted as other class")
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
print("OK ->", OUT_PNG)
