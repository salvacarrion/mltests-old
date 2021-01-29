import os
import pandas as pd

# Vars

DATA_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo"
RAW_FILES = ["es-en-gma", "pt-en-gma"]
# RAW_FILES = ["test-gma-en2es", "test-gma-en2pt", "test-gma-es2en", "test-gma-pt2en"]


for fname in RAW_FILES:
    DATASET_BIOLOGICAL = os.path.join(DATA_PATH, "cleaned", "scielo-gma", fname + "-biological")
    DATASET_HEALTH = os.path.join(DATA_PATH, "cleaned", "scielo-gma", fname + "-health")
    SAVEPATH = os.path.join(DATA_PATH, "cleaned", "scielo-gma", fname + "-merged")

    # Read files
    print(f"Reading file... ({fname})")
    df_biological = pd.read_csv(os.path.join(DATA_PATH, f"{DATASET_BIOLOGICAL}.csv"))
    df_health = pd.read_csv(os.path.join(DATA_PATH, f"{DATASET_HEALTH}.csv"))

    # Add domains
    df_biological["domain"] = "biological"
    df_health["domain"] = "health"

    # Concat dataframes
    df = pd.concat([df_biological, df_health])

    # Save data
    df.to_csv(SAVEPATH + ".csv", index=False)
    print("File saved!")
    print("")

print("Done!")
