import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import matplotlib.ticker as tkr
import seaborn as sns
sns.set()

from paper1.code import utils

FOLDER_NAME = "unconstrained"
BASE_PATH = f"/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq-{FOLDER_NAME}/"

DATASETS = [
    "scielo_health_es_en",
    "scielo_health_pt_en",
    "scielo_biological_es_en",
    "scielo_biological_pt_en",
    "scielo_merged_es_en",
    "scielo_merged_pt_en"
    # "scielo_health_biological_es_en",
    # "scielo_health_biological_pt_en",
]
TOKENS = False
if TOKENS:
    TITLE = f"Number of tokens across splits"
    TOK = "tok_"
else:
    TITLE = f"Number of sentences across splits"
    TOK = ""

rows = []
for dataset in DATASETS:

    for split in ["train", "val", "test"]:
        if TOKENS:
            filename = f"{dataset}/clean/{split}.tok.clean.en"
        else:
            filename = f"{dataset}/raw/{split}.en"

        with open(os.path.join(BASE_PATH, filename), 'r') as f:
            lines = f.read().strip().split('\n')

            if TOKENS:
                lengths = sum([len(x.split(' ')) for x in lines])
            else:
                lengths = len(lines)

            text = dataset.replace("scielo_", "").replace("_", "-")
            rows.append([text, split, lengths])

# Create dataframe
df = pd.DataFrame(rows, columns=["Dataset", "Split", "Size"])
print(df)

# Save sizes
df.to_csv(f"../../data/{FOLDER_NAME}/split_sizes_{TOK}{FOLDER_NAME}.csv")
print("Data saved!")

# Print sizes
print(df)

# Draw a nested barplot by species and sex
g = sns.catplot(data=df, x="Dataset", y="Size", kind="bar", hue="Split", legend=False)
g.fig.set_size_inches(12, 8)

# properties
g.set(xlabel='Datasets', ylabel='Sizes')
plt.title(TITLE)

g.set_xticklabels(rotation=45, horizontalalignment="center")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(utils.human_format)
plt.legend(loc='upper right')
plt.tight_layout()

# Save figure
plt.savefig(f"../../data/{FOLDER_NAME}/images/split_sizes_{TOK}{FOLDER_NAME}.pdf")
plt.savefig(f"../../data/{FOLDER_NAME}/images/split_sizes_{TOK}{FOLDER_NAME}.jpg")
print("Figure saved!")

# Show plot
plt.show()
asd = 3
