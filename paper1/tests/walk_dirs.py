import os
import subprocess

VOCAB_SIZE = 32000
if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/scielo/fairseq"
    FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
    FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

DOMAINS = ["health", "biological", "merged"]

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for domain in DOMAINS:
        dataset = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset, "valset-bleu5-detok-etc", "test.tok.clean.en")

        with open(path, 'r') as f:
            a = f.read().strip()
            print(f"{dataset}: {len(a)}")

