import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = [
    {"Model": "Health", "Test domain": "Health", "lang": "es-en", "BLEU": 39.31},
    {"Model": "Health", "Test domain": "Biological", "lang": "es-en", "BLEU": 40.53},
    {"Model": "Health", "Test domain": "Merged", "lang": "es-en", "BLEU": 40.08},

    {"Model": "Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 27.07},
    {"Model": "Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 33.60},
    {"Model": "Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 30.28},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 40.20},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 43.84},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 42.13},

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

    {"Model": "Health", "Test domain": "Health", "lang": "pt-en", "BLEU": 39.52},
    {"Model": "Health", "Test domain": "Biological", "lang": "pt-en", "BLEU": 40.06},
    {"Model": "Health", "Test domain": "Merged", "lang": "pt-en", "BLEU": 39.82},

    {"Model": "Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 25.65},
    {"Model": "Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 32.40},
    {"Model": "Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 29.22},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 40.19},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 41.95},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 41.15},

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


# Select language
LANG = "pt-en"
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
plt.savefig(f"../../data/images/bleu_scores_{LANG}.pdf")
print("Figure saved!")

# Show plot
plt.show()
asd = 3


