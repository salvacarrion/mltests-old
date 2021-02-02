import os
import subprocess

if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/tests"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/Scielo/fairseq/tests"

VOCAB_SIZE = 32000
DOMAINS = ["health", "biological", "merged"]


def generate():
    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        for domain in DOMAINS:
            dataset = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"
            print(f"Evaluating model from: {dataset}...")

            # Paths
            model_path = os.path.abspath(os.path.join(BASE_PATH, f"../{dataset}/checkpoints/transformer/checkpoint_best.pt"))
            testset_path = os.path.abspath(os.path.join(BASE_PATH, f"../{dataset}"))
            src_dict = os.path.abspath(os.path.join(BASE_PATH, f"../{dataset}/data-bin/{dataset}/dict.{SRC_LANG}.txt"))
            tgt_dict = os.path.abspath(os.path.join(BASE_PATH, f"../{dataset}/data-bin/{dataset}/dict.{TRG_LANG}.txt"))

            for domain2 in DOMAINS:
                # Preprocess files
                print(f"\t- Pre-processing testset from: {domain2}...")
                output_path = os.path.join(BASE_PATH, f"{SRC_LANG}-{TRG_LANG}", domain2)
                subprocess.call(['sh', './scripts/3_preprocess-test.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, src_dict, tgt_dict, testset_path, output_path])

                # Generate them
                print(f"\t- Generating translations for: {domain2}...")
                subprocess.call(['sh', './scripts/5_generate.sh', SRC_LANG, TRG_LANG, domain2, model_path, output_path, dataset])

                return


if __name__ == "__main__":
    generate()
