import os
import subprocess

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"
VOCAB_SIZE = 32000

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["health", "biological", "merged"]:
        dataset = f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset)
        subprocess.call(['sh', './scripts/4_train.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, dataset, path])
