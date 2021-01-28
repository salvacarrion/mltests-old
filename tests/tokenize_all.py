import os
from sacremoses import MosesTokenizer, MosesPunctNormalizer, MosesTruecaser


SRC_LANG = "es"
TRG_LANG = "en"
BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/scielo_biological_es_en"

for split in ["train", "val", "test"]:
    for lang in [SRC_LANG, TRG_LANG]:
        mt = MosesTokenizer(lang=lang)
        mpn = MosesPunctNormalizer()

        # Reading file
        filename = f"{BASE_PATH}/{split}.{lang}"
        with open(filename, 'r') as f:
            text = f.read()

        # Tokenize text
        print("Tokenizing text...")
        text = mt.tokenize(text, return_str=True)

        # Normalize text
        print("Normalizing text...")
        text = mpn.normalize(text)

        # True case
        tcm_savepath = f'{BASE_PATH}/truecasemodel.{lang}'
        if not os.path.exists(tcm_savepath):
            print("Training truecase model...")
            mtr = MosesTruecaser()
            mtr.train(text, save_to=tcm_savepath)
        else:
            mtr = MosesTruecaser(tcm_savepath)
        print("Truecasing text...")
        text = mtr.truecase(text, return_str=True, use_known=True)

        # Write file
        savepath = f"{BASE_PATH}/{split}.tok.clean.{lang}"
        with open(savepath, 'w') as f:
            f.write(f)
            print("File written!")
        asd = 3
