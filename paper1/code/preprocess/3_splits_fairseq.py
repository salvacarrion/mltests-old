import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SUFFLE = True

BASEPATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo"

DATA_PATH_TR = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned/scielo-gma"
RAW_FILES_TR = ["es-en-gma-biological.csv", "es-en-gma-health.csv", "es-en-gma-merged.csv",
                "pt-en-gma-biological.csv", "pt-en-gma-health.csv", "pt-en-gma-merged.csv"]

DATA_PATH_TS = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned/testset_gma"
RAW_FILES_TS = [
                "test-gma-es2en-biological.csv", "test-gma-es2en-health.csv", "test-gma-es2en-merged.csv",
                "test-gma-pt2en-biological.csv", "test-gma-pt2en-health.csv", "test-gma-pt2en-merged.csv"
                ]
                # "test-gma-en2es-biological.csv",  "test-gma-en2es-health.csv",  "test-gma-en2es-merged.csv",
                #"test-gma-en2pt-biological.csv",  "test-gma-en2pt-health.csv",  "test-gma-en2pt-merged.csv"


def get_domain(fname):
    if "health" in fname:
        return "health"
    elif "biological" in fname:
        return "biological"
    elif "merged" in fname:
        return "merged"
    else:
        raise NotImplementedError("Unknown domain")


def get_langs(fname, istrain):
    # Get languages
    if istrain:  # Train
        tmp = fname.split('-')
        SRC_LANG, TRG_LANG = tmp[:2]
    else:  # Test
        tmp = fname.split('-')
        SRC_LANG, TRG_LANG = tmp[2].split("2")
    return SRC_LANG, TRG_LANG


for fname_tr, fname_ts in zip(RAW_FILES_TR, RAW_FILES_TS):
    # Read files
    print(f"Reading files... ({fname_tr} AND {fname_ts})")
    df_tr = pd.read_csv(os.path.join(DATA_PATH_TR, fname_tr))
    df_ts = pd.read_csv(os.path.join(DATA_PATH_TS, fname_ts))

    # Shuffle dataset
    if SUFFLE:
        np.random.seed(123)
        np.random.shuffle(df_tr.values)
        np.random.shuffle(df_ts.values)

    # Get domain
    domain_tr = get_domain(fname_tr)
    domain_ts = get_domain(fname_ts)
    assert domain_tr == domain_ts
    domain = domain_tr

    # Get languages
    SRC_LANG_TR, TRG_LANG_TR = get_langs(fname_tr, istrain=True)
    SRC_LANG_TS, TRG_LANG_TS = get_langs(fname_ts, istrain=False)
    assert SRC_LANG_TR == SRC_LANG_TS
    assert TRG_LANG_TR == TRG_LANG_TS
    SRC_LANG = SRC_LANG_TR
    TRG_LANG = TRG_LANG_TR

    # Create path if it doesn't exists
    savepath = os.path.join(BASEPATH, "fairseq", f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}", "raw")
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # Split (Train / Validation)
    test_size = min(5000, max(3000, int(len(df_tr)*0.03)))
    train, val = train_test_split(df_tr, test_size=test_size)
    print(f"Train: {len(train)}; Val={len(val)}")
    for ds, ds_name in [(train, "train"), (val, "val")]:
        # Save as txt
        for lang in [SRC_LANG, TRG_LANG]:
            # Save file (one per language
            filename = os.path.join(savepath, f"{ds_name}.{lang}")
            with open(filename, 'w') as f:
                lines = map(lambda x: x + '\n', ds[lang])
                f.writelines(lines)
            print(f"File saved! ({filename})")

    # Test set
    for ds, ds_name in [(df_ts, "test")]:
        # Save as txt
        for lang in [SRC_LANG, TRG_LANG]:
            # Save file (one per language
            filename = os.path.join(savepath, f"{ds_name}.{lang}")
            with open(filename, 'w') as f:
                lines = map(lambda x: x + '\n', ds[lang])
                f.writelines(lines)
            print(f"File saved! ({filename})")

    print("Done!")
