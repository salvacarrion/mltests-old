import os
import subprocess

VOCAB_SIZE = 32000
# if os.environ.get('MACHINE') == "HOME":
print("Local")
BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq-constrained"
FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
# else:
#     print("Remote")
#     BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
#     FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

DOMAINS = ["health", "biological", "merged"]
EVALUATION_NAME = "evaluate_test_bleu5__seq2"


def walk():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        for domain1 in DOMAINS:
            dataset = f"scielo_{domain1}_{SRC_LANG}_{TRG_LANG}"
            print(f"{dataset}: *********************************")
            for domain2 in DOMAINS:
                path = os.path.join(BASE_PATH, EVALUATION_NAME, dataset, domain2, "eval", "generate-test.txt")

                with open(path, 'r') as f:
                    tmp = f.read().strip().split("\n")
                    print(f"\t- {domain2}:\n{tmp[-1]}")


def walk_finetune():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        dataset = f"scielo_health_biological_{SRC_LANG}_{TRG_LANG}"
        print(f"{dataset}: *********************************")
        for domain2 in DOMAINS:
            path = os.path.join(BASE_PATH, EVALUATION_NAME, dataset, domain2, "eval", "generate-test.txt")

            with open(path, 'r') as f:
                tmp = f.read().strip().split("\n")
                print(f"\t- {domain2}:\n{tmp[-1]}")


if __name__ == "__main__":
    walk_finetune()
