import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Vars
DOMAINS = ["health", "biological", "merged"]
DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/scielo"
SUFFLE = True

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for domain in DOMAINS:
        DATASET = os.path.join(DATA_PATH, "cleaned/testset-gma", f"{SRC_LANG}-{TRG_LANG}", f"test-gma-{SRC_LANG}2{TRG_LANG}-{domain}.csv")
        SAVEPATH = os.path.join(DATA_PATH, "fairseq", f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}", "raw")

        # Read files
        print(f"Reading file... ({DATASET})")
        df = pd.read_csv(DATASET)

        # Shuffle dataset
        if SUFFLE:
            np.random.seed(123)
            np.random.shuffle(df.values)

        # Create path if it doesn't exists
        Path(SAVEPATH).mkdir(parents=True, exist_ok=True)

        # Test set
        for ds, ds_name in [(df, "test")]:
            # Save as txt
            for lang in [SRC_LANG, TRG_LANG]:
                # Save file (one per language
                filename = os.path.join(SAVEPATH, f"{ds_name}.{lang}")
                with open(filename, 'w') as f:
                    lines = map(lambda x: x + '\n', ds[lang])
                    f.writelines(lines)
                print(f"File saved! ({filename})")

        # Split (Train / Validation)
        # test_size = min(5000, max(3000, int(len(df)*0.03)))
        # train, val = train_test_split(df, test_size=test_size)
        # print(f"Train: {len(train)}; Val={len(val)}")
        # for ds, ds_name in [(train, "train"), (val, "val")]:
        #
        #     # Save as txt
        #     for lang in [SRC_LANG, TRG_LANG]:
        #         # Save file (one per language
        #         filename = os.path.join(SAVEPATH, f"{ds_name}.{lang}")
        #         with open(filename, 'w') as f:
        #             lines = map(lambda x: x + '\n', ds[lang])
        #             f.writelines(lines)
        #         print(f"File saved! ({filename})")

        print("Done!")
