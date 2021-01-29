import os
import pandas as pd

from paper1.code import utils

# Vars

DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo"
# RAW_FILES = ["es-en-gma-biological", "es-en-gma-health", "fr-en-gma-health", "pt-en-gma-biological", "pt-en-gma-health"]
RAW_FILES = ["test-gma-en2es-biological", "test-gma-en2es-health", "test-gma-en2fr-health", "test-gma-en2pt-biological", "test-gma-en2pt-health", "test-gma-es2en-biological", "test-gma-es2en-health", "test-gma-fr2en-health", "test-gma-pt2en-biological", "test-gma-pt2en-health",]


for fname in RAW_FILES:
    DATASET = os.path.join(DATA_PATH, "raw", "testset-gma", fname)
    SAVEPATH = os.path.join(DATA_PATH, "cleaned", "testset-gma", fname)

    # Get languages (train)
    # tmp = fname.split('-')
    # SRC_LANG, TRG_LANG = tmp[:2]

    # Get languages (test)
    tmp = fname.split('-')
    SRC_LANG, TRG_LANG = tmp[2].split("2")

    # Read files
    print(f"Reading file... ({fname})")
    filename = os.path.join(DATA_PATH, f"{DATASET}.csv")
    df = pd.read_csv(filename)

    # Clean dataset
    total_old = len(df)
    df = utils.preprocess_dataset(df, src_col=SRC_LANG, trg_col=TRG_LANG)

    # Stats
    total_doctypes = df['doctype'].value_counts()
    removed = total_old-len(df)
    print(f"Stats for: {fname} **************************")
    print(f"\t- Documents: {len(set(df['docid']))}")
    print(f"\t- Sentences: {len(df)}")
    print("\t\t- Removed: {} ({:.2f}%)".format(removed, removed/total_old*100))
    print("\t- Titles/Abstracts: {}/{} ({:.2f}%)".format(total_doctypes['title'], total_doctypes['text'], total_doctypes['title']/total_doctypes['text']*100))

    # Save data
    df.to_csv(SAVEPATH + ".csv", index=False)
    print("File saved!")
    print("")

print("Done!")
