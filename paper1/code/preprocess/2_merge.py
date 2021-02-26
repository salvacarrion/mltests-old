import os
import pandas as pd

TRAIN = True
if TRAIN:
    DATA_PATH = SAVEPATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned-constrained/scielo-gma"
    RAW_FILES = [("es-en-gma-biological.csv", "es-en-gma-health.csv"),
                 ("pt-en-gma-biological.csv", "pt-en-gma-health.csv")]
else:
    DATA_PATH = SAVEPATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned-constrained/testset_gma"
    RAW_FILES = [("test-gma-en2es-biological.csv", "test-gma-en2es-health.csv"),
                 ("test-gma-en2pt-biological.csv", "test-gma-en2pt-health.csv"),
                 ("test-gma-es2en-biological.csv", "test-gma-es2en-health.csv"),
                 ("test-gma-pt2en-biological.csv", "test-gma-pt2en-health.csv")]

for fname1, fname2 in RAW_FILES:
    # Read files
    print(f"Reading files... ({fname1} AND {fname2})")
    df_file1 = pd.read_csv(os.path.join(DATA_PATH, fname1))
    df_file2 = pd.read_csv(os.path.join(DATA_PATH, fname2))

    # Add domains
    df_file1["domain"] = "health" if "health" in fname1 else "biological"
    df_file2["domain"] = "health" if "health" in fname2 else "biological"

    # Concat dataframes
    df = pd.concat([df_file1, df_file2])

    # Save data
    save_fname = "-".join(fname1.split('-')[:-1]) + "-merged.csv"
    df.to_csv(os.path.join(SAVEPATH, save_fname), index=False)
    print("File saved!")
    print("")

print("Done!")
