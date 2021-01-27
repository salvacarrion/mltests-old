import os
import pandas as pd
import tqdm
import json
import numpy as np
from pathlib import Path

from paper1 import utils

# Vars
SRC_LANG = "pt"
TRG_LANG = "en"
SPLIT_NAME = "test"
DOMAIN = "merged"
DATASET_NAME = f"scielo_{DOMAIN}"
FOLDER_NAME = f"{DATASET_NAME}_{SRC_LANG}_{TRG_LANG}"

DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo"
DATASET = [f"test-gma-{SRC_LANG}2{TRG_LANG}-{DOMAIN}", f"test-gma-{TRG_LANG}2{SRC_LANG}-{DOMAIN}"]
SAVEPATH = os.path.join(DATA_PATH, "fairseq", FOLDER_NAME)

SUFFLE = True

for d in DATASET:
    # Read files
    print(f"Reading file... ({d})")
    df = pd.read_csv(os.path.join(DATA_PATH, f"cleaned/testset-gma/{d}.csv"))

    # Shuffle dataset
    if SUFFLE:
        np.random.seed(123)
        np.random.shuffle(df.values)

    # Create path if it doesn't exists
    Path(SAVEPATH).mkdir(parents=True, exist_ok=True)

    # Save as txt
    for lang in [SRC_LANG, TRG_LANG]:
        # Save file (one per language
        filename = os.path.join(SAVEPATH, f"{SPLIT_NAME}_{DATASET_NAME}.{lang}")
        with open(filename, 'w') as f:
            lines = map(lambda x: x + '\n', df[lang])
            f.writelines(lines)
        print(f"File saved! ({filename})")

    print("Done!")
