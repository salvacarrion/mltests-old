import os
import bioc
import pandas as pd
import tqdm

# Vars
SRC_LANG = "pt"
TRG_LANG = "en"
DATA_PATH = "/Users/salvacarrion/Documents/Programming/data/wmt16biomedical"
DATASET = "scielo-gma/pt-en-gma-health"
SAVEPATH = os.path.join(DATA_PATH, "preprocessed", DATASET)

# Read files
print("Reading files...")
DIR1 = os.path.join(DATA_PATH, DATASET)
filenames = [file for file in os.listdir(DIR1) if file.endswith(".txt")]

# Process files
print("Processing files...")
data = {}
for i, fname in tqdm.tqdm(enumerate(filenames), total=len(filenames)):
    # Extract parts
    values = fname.split("_")

    # Get parts
    docid = values[0]
    doctype = values[1].lower()
    lang = SRC_LANG if SRC_LANG.lower() in values[2].lower() else TRG_LANG

    # Read file
    with open(os.path.join(DIR1, fname), 'r') as f:
        text = f.read()

    # Rows
    key = f"{docid}-{doctype}"
    if key not in data:
        data[key] = {"docid": docid, "doctype": doctype}
    data[key][lang] = text

    # For debugging
    # if i+1 >= 100:
    #     break

# Save data
df = pd.DataFrame(data=data.values())
df.to_csv(SAVEPATH + ".csv", index=False)
print("File saved!")

# Check values  (save first)
assert min([len(v) for k, v in data.items()]) == 4

print("Done!")

