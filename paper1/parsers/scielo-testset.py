import os
import bioc
import pandas as pd
import tqdm
import itertools

# Vars
DATA_PATH = "/Users/salvacarrion/Documents/Programming/data/wmt16biomedical"

dirs = ["test-gma-en2es-biological", "test-gma-en2es-health", "test-gma-en2fr-health", "test-gma-en2pt-biological", "test-gma-en2pt-health",
        #"test-gma-es2en-biological", "test-gma-es2en-health", "test-gma-fr2en-health", "test-gma-pt2en-biological", "test-gma-pt2en-health"]
        ]
for cdir in dirs:
    DATASET = f"testset_gma/{cdir}"
    SAVEPATH = os.path.join(DATA_PATH, "preprocessed", DATASET)

    langs = cdir.split('-')[2]
    SRC_LANG, TRG_LANG = langs.split("2")

    # Read files
    print("Reading files...")
    DIR1 = os.path.join(DATA_PATH, DATASET)
    filenames = [file for file in os.listdir(DIR1) if file.endswith(".crp")]

    # Process files
    print("Processing files...")
    data = []
    for i, fname in tqdm.tqdm(enumerate(filenames), total=len(filenames)):
        # Read file
        with open(os.path.join(DIR1, fname), 'r') as f:
            lines = f.readlines()

        # Process lines
        for i in range(0, len(lines), 3):
            cid, src, trg = lines[i:i+3]
            docid, doctype = cid.split('_')
            row = {"docid": docid.strip(), "doctype": doctype.lower().strip(), SRC_LANG: src.strip(), TRG_LANG: trg.strip()}
            data.append(row)

        # # For debugging
        # if i+1 >= 100:
        #     break

    # Save data
    df = pd.DataFrame(data=data)
    df.to_csv(SAVEPATH + ".csv", index=False)
    print("File saved!")

    # Check values  (save first)
    assert min([len(d) for d in data]) == 4

print("Done!")

