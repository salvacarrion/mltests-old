from seq2seq.mt.transformer_sys import LitTokenizer


# Constants
DATASET_PATH = f"../.data/miguel"
SRC_LANG, TRG_LANG = ("en", "es")

# Define Tokenizer
# Do not use padding here. Datasets are preprocessed before batching
tokenizer = LitTokenizer(padding=False, truncation=False)

# Train tokenizer
src_files = [f"{DATASET_PATH}/preprocessed/train_{SRC_LANG}.csv"]
trg_files = [f"{DATASET_PATH}/preprocessed/train_{TRG_LANG}.csv"]
tokenizer.train_vocabs(src_files, trg_files)
print("Training done!")

# Save vocabs
folder = f"{DATASET_PATH}/vocab"
tokenizer.save_vocabs(folder, src_name=SRC_LANG, trg_name=TRG_LANG)
print("Vocabs saved!")
