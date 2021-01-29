import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Vars
SRC_LANG = "es"
TRG_LANG = "en"
DOMAINS = ["health", "biological", "merged"]

for DOMAIN in DOMAINS:
    DATASET_NAME = f"scielo_{DOMAIN}"
    FOLDER_NAME = f"{DATASET_NAME}_{SRC_LANG}_{TRG_LANG}"

    DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo"
    DATASET = [f"{SRC_LANG}-{TRG_LANG}-gma-{DOMAIN}"]
    SAVEPATH = os.path.join(DATA_PATH, "fairseq", FOLDER_NAME)

    SUFFLE = True

    for d in DATASET:
        # Read files
        print(f"Reading file... ({d})")
        df = pd.read_csv(os.path.join(DATA_PATH, f"cleaned/scielo-gma/{d}.csv"))

        # Shuffle dataset
        if SUFFLE:
            np.random.seed(123)
            np.random.shuffle(df.values)

        # Create path if it doesn't exists
        Path(SAVEPATH).mkdir(parents=True, exist_ok=True)

        # Split
        test_size = min(5000, max(3000, int(len(df)*0.03)))
        train, val = train_test_split(df, test_size=test_size)
        print(f"Train: {len(train)}; Val={len(val)}")
        for ds, ds_name in [(train, "train"), (val, "val")]:

            # Save as txt
            for lang in [SRC_LANG, TRG_LANG]:
                # Save file (one per language
                filename = os.path.join(SAVEPATH, f"{ds_name}.{lang}")
                with open(filename, 'w') as f:
                    lines = map(lambda x: x + '\n', ds[lang])
                    f.writelines(lines)
                print(f"File saved! ({filename})")

        print("Done!")
