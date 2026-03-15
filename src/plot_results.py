import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import RESULTS_DIR,BASE_DIR
sns.set(style="whitegrid")

df = pd.read_csv(os.path.join(RESULTS_DIR,"summary.csv"))

os.makedirs(os.path.join(BASE_DIR,"figures"), exist_ok=True)


plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="strategy",
    y="strict_recall",
    hue="retriever"
)

plt.title("Strict Recall by Strategy and Retriever")
plt.ylabel("Strict Recall")
plt.xlabel("Query Rewriting Strategy")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR,"figures","recall_by_retriever.png"))
plt.close()



plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="strategy",
    y="strict_recall",
    hue="dataset"
)

plt.title("Strict Recall Across Datasets")
plt.ylabel("Strict Recall")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR,"figures","recall_by_dataset.png"))
plt.close()



plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="strategy",
    y="hallucination_rate"
)

plt.title("Hallucination Rate by Strategy")
plt.ylabel("Hallucination Rate")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR,"figures","hallucination_rate.png"))
plt.close()

print("Figures saved to figures/")