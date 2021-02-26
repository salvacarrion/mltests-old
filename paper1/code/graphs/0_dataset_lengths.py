import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"
FOLDER_NAME = "constrained"

DATASETS = [
    "scielo_biological_es_en",
    "scielo_biological_pt_en",
    # "scielo_health_biological_es_en",
    # "scielo_health_biological_pt_en",
    "scielo_health_es_en",
    "scielo_health_pt_en",
    "scielo_merged_es_en",
    "scielo_merged_pt_en"
]

rows = []
for dataset in DATASETS:

    for split in ["train", "val", "test"]:
        filename = f"{dataset}/raw/{split}.en"
        with open(os.path.join(BASE_PATH, filename), 'r') as f:
            length = len(f.read().strip().split('\n'))
            text = dataset.replace("scielo_", "").replace("_", "-")
            rows.append([text, split, length])

# Save sizes
df = pd.DataFrame(rows, columns=["Dataset", "Split", "Size"])
df.to_csv(f"../../data/{FOLDER_NAME}/split_sizes_{FOLDER_NAME}.csv")
print("Data saved!")

# Print sizes
print(df)

# Draw a nested barplot by species and sex
g = sns.catplot(data=df, x="Dataset", y="Size", kind="bar", hue="Split", legend=False)
g.fig.set_size_inches(12, 8)

# properties
g.set(xlabel='Datasets', ylabel='Sizes')
plt.title(f"Sizes across splits ({FOLDER_NAME})")

g.set_xticklabels(rotation=45, horizontalalignment="center")
plt.legend(loc='upper right')
plt.tight_layout()

# Save figure
plt.savefig(f"../../data/{FOLDER_NAME}/images/split_sizes_{FOLDER_NAME}.pdf")
plt.savefig(f"../../data/{FOLDER_NAME}/images/split_sizes_{FOLDER_NAME}.jpg")
print("Figure saved!")

# Show plot
plt.show()
asd = 3
