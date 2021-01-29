import subprocess

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["health", "biological", "merged"]:
        subprocess.call(['sh', './scripts/1_tokenize.sh', SRC_LANG, TRG_LANG, BASE_PATH + f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"])
