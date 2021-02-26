import os
import pandas as pd
from pathlib import Path
import numpy as np
from paper1.code import utils


TRAIN = True
SUFFLE = True
CONSTRAINED = True

if TRAIN:
    # Train
    SAVEPATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned-constrained/scielo-gma"
    DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/originals/scielo-gma/scielo-gma"
    RAW_FILES = ["es-en-gma-biological.csv", "es-en-gma-health.csv", "fr-en-gma-health.csv", "pt-en-gma-biological.csv", "pt-en-gma-health.csv"]
    print("Processing training files...")
else:
    # Test
    SAVEPATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned-constrained/testset_gma"
    DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/originals/testset-gma/testset_gma"
    RAW_FILES = ["test-gma-en2es-biological.csv", "test-gma-en2es-health.csv", "test-gma-en2fr-health.csv", "test-gma-en2pt-biological.csv", "test-gma-en2pt-health.csv", "test-gma-es2en-biological.csv", "test-gma-es2en-health.csv", "test-gma-fr2en-health.csv", "test-gma-pt2en-biological.csv", "test-gma-pt2en-health.csv"]
    print("Processing test files...")


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

# Create path if doesn't exists
path = Path(SAVEPATH)
path.mkdir(parents=True, exist_ok=False)

for fname in RAW_FILES:
    # Read file
    print(f"Reading file... ({fname})")
    filename = os.path.join(DATA_PATH, fname)
    df = pd.read_csv(filename)

    # Limit dataset
    domain = get_domain(fname)
    SRC_LANG, TRG_LANG = get_langs(fname, istrain=TRAIN)

    # Clean dataset
    total_old = len(df)
    df = utils.preprocess_dataset(df, src_col=SRC_LANG, trg_col=TRG_LANG)

    # Shuffle dataset
    if SUFFLE:
        np.random.seed(123)
        np.random.shuffle(df.values)

    if CONSTRAINED and TRAIN:
        if domain == "health" and "es" in {SRC_LANG, TRG_LANG}:
            max_size = 123597
            print(f"Limiting size to {max_size}")
            df = df[:max_size]
        elif domain == "health" and "pt" in {SRC_LANG, TRG_LANG}:
            max_size = 120301
            print(f"Limiting size to {max_size}")
            df = df[:max_size]

    # Stats
    total_doctypes = df['doctype'].value_counts()
    removed = total_old-len(df)
    print(f"Stats for: {fname} **************************")
    print(f"\t- Documents: {len(set(df['docid']))}")
    print(f"\t- Sentences: {len(df)}")
    print("\t\t- Removed: {} ({:.2f}%)".format(removed, removed/total_old*100))
    print("\t- Titles/Abstracts: {}/{} ({:.2f}%)".format(total_doctypes['title'], total_doctypes['text'], total_doctypes['title']/total_doctypes['text']*100))

    # Save data
    df.to_csv(os.path.join(SAVEPATH, fname), index=False)
    print("File saved!")
    print("")

print("Done!")
