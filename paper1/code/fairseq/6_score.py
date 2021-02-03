import os
import subprocess

if os.environ.get('MACHINE') == "HOME":
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/scielo/fairseq/"
else:
    BASE_PATH = "/home/scarrion/datasets/Scielo/fairseq/"

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for d in ["health", "biological", "merged"]:
        for model in ["transformer/"]:
            dataset = f"scielo_{d}_{SRC_LANG}_{TRG_LANG}"
            path = os.path.join(BASE_PATH, dataset)
            subprocess.call(['sh', './scripts/5_evaluate.sh', SRC_LANG, TRG_LANG, model+"checkpoint_best.pt", dataset, path])
