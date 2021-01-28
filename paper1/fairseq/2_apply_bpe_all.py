import subprocess

VOCAB_SIZE = 32000
BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["health", "biological", "merged"]:
        subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, BASE_PATH+f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"])
