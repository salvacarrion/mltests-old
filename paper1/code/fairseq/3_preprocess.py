import os
import subprocess

import sys

if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq"
    FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
    FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

VOCAB_SIZE = 32000


def train():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        for domain in ["health", "biological", "merged"]:
            dataset = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"

            path = os.path.join(BASE_PATH, dataset)
            subprocess.call(['sh', './scripts/3_preprocess.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, dataset, path])


def train_finetune():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        dataset = f"scielo_health_biological_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset)
        subprocess.call(['sh', './scripts/3_preprocess-finetune.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, dataset, path])


if __name__ == "__main__":
    train_finetune()
