import os
import bioc
import pandas as pd
import tqdm
import json

# Vars
SRC_LANG = "en"
TRG_LANG = "pt"
DATA_PATH = "/Users/salvacarrion/Documents/Programming/data/wmt16biomedical"
DATASET = f"medline/pubmed_{SRC_LANG}_{TRG_LANG}"
SAVEPATH = os.path.join(DATA_PATH, "preprocessed", DATASET)

# Read files
print("Parsing file...")
filename = os.path.join(DATA_PATH, f"{DATASET}.txt")
data = []
with open(filename, 'r') as fp:
    lines = fp.readlines()
    for line in tqdm.tqdm(lines, total=len(lines)):
        docid, src, trg = line.split('|')
        data.append({"docid": docid, SRC_LANG: src.strip(), TRG_LANG: trg.strip()})

# Save data
df = pd.DataFrame(data=data)
df.to_csv(SAVEPATH + ".csv", index=False)
print("File saved!")

# Check values  (save first)
assert min([len(d) for d in data]) == 3

print("Done!")


