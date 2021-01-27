import os
import bioc
import pandas as pd
import tqdm
import json

# Vars
DATA_PATH = "/Users/salvacarrion/Documents/Programming/datasets/wmt16biomedical"


# # Load data (debugging)
# with open(SAVEPATH + "_unaligned.json", 'r') as f:
#     data = json.load(f)
#     print("File loaded!")
dirs= ["en-biological", "en-health", "es-biological", "es-health", "fr-health", "pt-biological", "pt-health"]
for d in dirs:
    DATASET = f"scielo-monolingual/{d}"
    SAVEPATH = os.path.join(DATA_PATH, "preprocessed", DATASET)

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

                key = f"{docid}-{doctype}-{sentnum}"
                if key not in data:
                    data[key] = {"docid": docid, "doctype": doctype, "sentnum": sentnum}

                data[key][lang] = text

    # Save data
    df = pd.DataFrame(data=data.values())
    df.to_csv(SAVEPATH + ".csv", index=False)
    print("File saved!")

    # Check values  (save first)
    assert min([len(d) for d in data.values()]) == 4

print("Done!")

