import os
import bioc
import pandas as pd
import tqdm
import json

# Vars
DATA_PATH = "/Users/salvacarrion/Documents/Programming/data/wmt16biomedical"
DATASET = "scielo/pt-en-training-health"
SAVEPATH = os.path.join(DATA_PATH, "preprocessed", DATASET)

# # Load data (debugging)
# with open(SAVEPATH + "_unaligned.json", 'r') as f:
#     data = json.load(f)
#     print("File loaded!")

# Read files
print("Reading file...")
filename = os.path.join(DATA_PATH, f"{DATASET}.xml")
with open(filename, 'r') as fp:
    collection = bioc.load(fp)

# Process files
print("Processing files...")
data = {}
for i, doc in tqdm.tqdm(enumerate(collection.documents), total=len(collection.documents)):
    docid = doc.id

    # Parse passages
    for passage in doc.passages:
        doctype = passage.infons['section'].lower()
        lang = passage.infons['language'].lower()

        # Parse sentences
        for sent in passage.sentences:
            sentnum = sent.infons["sentnum"]
            text = sent.text

            key = f"{docid}-{doctype}"
            if key not in data:
                data[key] = {"docid": docid, "doctype": doctype}

            if lang not in data[key]:
                data[key][lang] = {}
            data[key][lang][sentnum] = text

# Save data
with open(SAVEPATH + "_unaligned.json", 'w') as f:
    json.dump(data, f)
    print("File saved!")

print("Done!")

