import os

BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/"
VOCAB_SIZE = 32000

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for DOMAIN in ["health", "biological", "merged"]:

        # Read file
        filename = f"scielo_{DOMAIN}_{SRC_LANG}_{TRG_LANG}"
        with open(os.path.join(BASE_PATH, filename), 'r') as f:
            codes = f.read()