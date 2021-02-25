import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = [
    {"Model": "Health", "Test domain": "Health", "lang": "es-en", "BLEU": 39.02},
    {"Model": "Health", "Test domain": "Biological", "lang": "es-en", "BLEU": 39.38},
    {"Model": "Health", "Test domain": "Merged", "lang": "es-en", "BLEU": 39.34},

    {"Model": "Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 26.19},
    {"Model": "Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 32.76},
    {"Model": "Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 29.36},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 39.63},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 42.95},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 41.32},

    {"Model": "Health→Biological\n(Naive)", "Test domain": "Health", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Naive)", "Test domain": "Biological", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Naive)", "Test domain": "Merged", "lang": "es-en", "BLEU": 1},

    {"Model": "Health→Biological\n(EWC)", "Test domain": "Health", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(EWC)", "Test domain": "Biological", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(EWC)", "Test domain": "Merged", "lang": "es-en", "BLEU": 1},

    {"Model": "Health→Biological\n(Ours)", "Test domain": "Health", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Ours)", "Test domain": "Biological", "lang": "es-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Ours)", "Test domain": "Merged", "lang": "es-en", "BLEU": 1},
    
    
    ##

    {"Model": "Health", "Test domain": "Health", "lang": "pt-en", "BLEU": 38.86},
    {"Model": "Health", "Test domain": "Biological", "lang": "pt-en", "BLEU": 39.61},
    {"Model": "Health", "Test domain": "Merged", "lang": "pt-en", "BLEU": 39.26},

    {"Model": "Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 25.67},
    {"Model": "Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 31.94},
    {"Model": "Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 28.86},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 39.54},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 41.80},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 40.74},

    {"Model": "Health→Biological\n(Naive)", "Test domain": "Health", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Naive)", "Test domain": "Biological", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Naive)", "Test domain": "Merged", "lang": "pt-en", "BLEU": 1},

    {"Model": "Health→Biological\n(EWC)", "Test domain": "Health", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(EWC)", "Test domain": "Biological", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(EWC)", "Test domain": "Merged", "lang": "pt-en", "BLEU": 1},

    {"Model": "Health→Biological\n(Ours)", "Test domain": "Health", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Ours)", "Test domain": "Biological", "lang": "pt-en", "BLEU": 1},
    {"Model": "Health→Biological\n(Ours)", "Test domain": "Merged", "lang": "pt-en", "BLEU": 1},
]

df = pd.DataFrame(data, columns=["Model", "Test domain", "lang", "BLEU"])
df.to_csv(f"../../data/test_data.csv")

# Select language
LANG = "es-en"
df = df[df.lang == LANG]

# Draw a nested barplot by species and sex
g = sns.catplot(data=df, x="Model", y="BLEU", kind="bar", hue="Test domain", legend=False)
g.fig.set_size_inches(12, 8)

# properties
g.set(xlabel='Models', ylabel='BLEU')
plt.title(f"BLUE scores in different domains | {LANG}")

g.set_xticklabels(rotation=0, horizontalalignment="center")
plt.legend(loc='upper right')
plt.tight_layout()

# Save figure
plt.savefig(f"../../data/images/bleu_scores_{LANG}__2.pdf")
print("Figure saved!")

# Show plot
plt.show()
asd = 3


