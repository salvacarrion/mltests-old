import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = [
    {"Model": "Health", "Test domain": "Health", "lang": "es-en", "BLEU": 39.14},
    {"Model": "Health", "Test domain": "Biological", "lang": "es-en", "BLEU": 39.52},
    {"Model": "Health", "Test domain": "Merged", "lang": "es-en", "BLEU": 39.45},

    {"Model": "Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 26.37},
    {"Model": "Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 32.81},
    {"Model": "Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 29.60},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "es-en", "BLEU": 39.78},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "es-en", "BLEU": 42.74},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "es-en", "BLEU": 41.34},

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

    {"Model": "Health", "Test domain": "Health", "lang": "pt-en", "BLEU": 38.79},
    {"Model": "Health", "Test domain": "Biological", "lang": "pt-en", "BLEU": 39.51},
    {"Model": "Health", "Test domain": "Merged", "lang": "pt-en", "BLEU": 39.18},

    {"Model": "Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 25.68},
    {"Model": "Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 31.68},
    {"Model": "Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 28.74},

    {"Model": "Health+Biological", "Test domain": "Health", "lang": "pt-en", "BLEU": 39.79},
    {"Model": "Health+Biological", "Test domain": "Biological", "lang": "pt-en", "BLEU": 41.56},
    {"Model": "Health+Biological", "Test domain": "Merged", "lang": "pt-en", "BLEU": 40.72},

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
plt.savefig(f"../../data/images/bleu_scores_{LANG}__2.pdf")
print("Figure saved!")

# Show plot
plt.show()
asd = 3


