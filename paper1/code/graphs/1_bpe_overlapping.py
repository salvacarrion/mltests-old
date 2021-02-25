import os
import pandas as pd

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"
VOCAB_SIZE = 32000
LANG_PAIRS = [("es", "en"), ("pt", "en")]
DOMAINS = ["health", "biological", "merged"]
rows = []

for SRC_LANG, TRG_LANG in LANG_PAIRS:
    vocabs = {}

    for domain in DOMAINS:
        vocabs[domain] = {}

        # Read vocabs
        for lang in [SRC_LANG, TRG_LANG]:
            vocab_filename = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}/bpe/train.tok.bpe.32000.{lang}"
            with open(os.path.join(BASE_PATH, vocab_filename), 'r') as f:
                vocabs[domain][lang] = {w.split(' ')[0].strip() for w in f.read().strip().split('\n')}

        # # Read target vocab
        # vocab_trg_filename = f"scielo_{DOMAIN}_{SRC_LANG}_{TRG_LANG}/vocab.tok.bpe.32000.{TRG_LANG}"
        # with open(os.path.join(BASE_PATH, vocab_trg_filename), 'r') as f: trg_vocab = f.read()
        #

    # Compute Overlap
    for lang in [SRC_LANG, TRG_LANG]:
        for domain1 in DOMAINS:
            vocab1 = vocabs[domain1][lang]

            for domain2 in DOMAINS:
                vocab2 = vocabs[domain2][lang]

                # Compute overlap
                intersection = len(vocab1.intersection(vocab2))
                union = len(vocab1.union(vocab2))
                iou = intersection/union
                overlap = intersection/len(vocab1)

                # Add row
                row = {"dataset": f"{SRC_LANG}-{TRG_LANG}", "lang": lang, "domain1": domain1, "domain2": domain2, "overlap": overlap, "iou": iou, "intersection": intersection, "union": union}
                rows.append(row)

# Create pandas and print
df = pd.DataFrame(data=rows)
print(df)

# Save file
df.to_csv("../../data/overlapping.csv", index=False)
print("File saved!")
