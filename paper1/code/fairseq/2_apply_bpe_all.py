import os
import subprocess

VOCAB_SIZE = 32000
if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq"
    FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
    FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

DOMAINS = ["health", "biological", "merged"]


def train():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        for domain in ["health", "biological", "merged"]:
            dataset = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"

            path = os.path.join(BASE_PATH, dataset)
            subprocess.call(['sh', './scripts/2_learn_bpe.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, path, FAST_PATH])
            subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, path, FAST_PATH])


def train_finetune():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
            dataset = f"scielo_health_biological_{SRC_LANG}_{TRG_LANG}"

            path = os.path.join(BASE_PATH, dataset)
            subprocess.call(['sh', './scripts/2_apply_bpe.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, path, FAST_PATH])


if __name__ == "__main__":
    train()
