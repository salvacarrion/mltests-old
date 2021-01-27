import os
import bioc
import pandas as pd
import tqdm
import json

# Vars
DATA_PATH = "/Users/salvacarrion/Documents/Programming/data/wmt16biomedical"
SAVEPATH = os.path.join(DATA_PATH, "preprocessed", "testset")

DATASET = "biological_en2es"
DATASET_TESTSET = f"testset/{DATASET}"
DATASET_GOLD = f"testset_gold/{DATASET}"


def read_file(filename, data=None):
    if data is None:
        data = {}

    # Read files
    print("Reading file...")
    with open(filename, 'r') as fp:
        collection = bioc.load(fp)

    # Process files
    print("Processing files...")
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

                # Add key if new
                key = f"{docid}-{doctype}-{sentnum}"
                if key not in data:
                    data[key] = {"docid": docid, "doctype": doctype, "sentnum": sentnum}

                # Add tests
                data[key][lang] = text

    return data

# Read test: source
filename_test = os.path.join(DATA_PATH, f"{DATASET_TESTSET}.xml")
data = read_file(filename_test)

# Read test: references
filename_test = os.path.join(DATA_PATH, f"{DATASET_GOLD}.xml")
data = read_file(filename_test, data)

# Save data
df = pd.DataFrame(data=data)
df.to_csv(SAVEPATH + ".csv", index=False)
print("File saved!")

# Check values  (save first)
assert min([len(v) for k, v in data.items()]) == 5

print("Done!")

