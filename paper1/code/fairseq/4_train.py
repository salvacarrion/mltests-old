import os
import subprocess

if os.environ.get('MACHINE') == "HOME":
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"
else:
    BASE_PATH = "/home/scarrion/datasets/Scielo/fairseq/"

VOCAB_SIZE = 32000

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["biological"]:
        dataset = f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset)
        subprocess.call(['sh', './scripts/4_train.sh', SRC_LANG, TRG_LANG, path])
