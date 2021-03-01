import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt

LANG_PAIRS = [("es", "en"), ("pt", "en")]
DOMAINS = ["health", "biological", "merged"]
FOLDER_NAME = "constrained"
# Read data
df = pd.read_csv(f"../../data/{FOLDER_NAME}/overlapping.csv")
col="iou" #"overlap"  #iou
# Plot
for src_lang, trg_lang in LANG_PAIRS:
    key = f"{src_lang}-{trg_lang}"

    for lang in [src_lang, trg_lang]:
        plt.figure()
        data = np.zeros((3,3))
        for i, domain1 in enumerate(DOMAINS):
            for j, domain2 in enumerate(DOMAINS):

                mask = (df["dataset"] == key) & (df["lang"] == lang) & (df["domain1"] == domain1) & (df["domain2"] == domain2)
                row = df[mask]
                value = float(row[col].values[0])
                data[i, j] = value

        heat_map = sb.heatmap(data, annot=True)
        # plt.title(f"IoU for '{lang}' (dataset: {key})")
        heat_map.set_xticklabels([x.title() for x in DOMAINS], ha='center', minor=False)
        heat_map.set_yticklabels([x.title() for x in DOMAINS], va='center', minor=False)
        plt.savefig(f"../../data/{FOLDER_NAME}/images/overlappig-{col}-{key}__{lang}.jpg", dpi=300)
        plt.savefig(f"../../data/{FOLDER_NAME}/images/overlappig-{col}-{key}__{lang}.pdf", dpi=300)
        print("File saved!")
        # plt.show()
        asd = 3

print("Done!")
