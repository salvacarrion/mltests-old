import os
import subprocess

VOCAB_SIZE = 32000
if os.environ.get('MACHINE') == "HOME":
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/scielo/fairseq/"
else:
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq/"

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["health", "biological", "merged"]:
        dataset = f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset)
        subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, path])
