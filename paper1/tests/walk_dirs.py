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
    for domain1 in DOMAINS:
        dataset = f"scielo_{domain1}_{SRC_LANG}_{TRG_LANG}"
        print(domain1)
        for domain2 in DOMAINS:
            path = os.path.join(BASE_PATH, "evaluate_test_bleu5", dataset, domain2, "eval", "generate-test.txt")

            with open(path, 'r') as f:
                a = f.read().strip().split("\n")
                print(f"\t- {domain2}:\n{a[-1]}")
                asds = 3
