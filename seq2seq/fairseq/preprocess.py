
import numpy as np
from scipy import stats
from datasets import load_dataset

import matplotlib.pyplot as plt

from seq2seq.fairseq.tknr import FairTokenizer

import pandas as pd

#%%

DATASET_PATH = f"../.data/miguel"
SRC_LANG, TRG_LANG = ("en", "es")

# Define Tokenizer
# Do not use padding here. Datasets are preprocessed before batching
src_tokenizer = FairTokenizer(lang="en", tokenizer="moses")
trg_tokenizer = FairTokenizer(lang="es", tokenizer="moses")

# Test tokenizer
print(src_tokenizer.tokenize("Transfomers are awesome! 😄"))
print(trg_tokenizer.tokenize("Los transfomers son geniales! 😄"))

# Test BPE
# x = src_tokenizer.apply_bpe([src_tokenizer.tokenize("Transfomers are awesome! 😄"), src_tokenizer.tokenize("Transgender are normal")])
# print(x)

def encode_src(x):
    clean = src_tokenizer.normalize(x)
    tokens = src_tokenizer.tokenize(clean)
    tokens_str = " ".join(tokens)
    return tokens_str


def encode_trg(x):
    clean = src_tokenizer.normalize(x)
    tokens = trg_tokenizer.tokenize(clean)
    tokens_str = " ".join(tokens)
    return tokens_str


# Read
print("Reading datasets...")
train_df = pd.read_csv(f"{DATASET_PATH}/preprocessed/train.csv")
dev_df = pd.read_csv(f"{DATASET_PATH}/preprocessed/dev.csv")
test_df = pd.read_csv(f"{DATASET_PATH}/preprocessed/test.csv")

# Encode datasets
print("Encoding datasets...")
train_df[SRC_LANG] = train_df[SRC_LANG].apply(encode_src)
train_df[TRG_LANG] = train_df[TRG_LANG].apply(encode_trg)
dev_df[SRC_LANG] = dev_df[SRC_LANG].apply(encode_src)
dev_df[TRG_LANG] = dev_df[TRG_LANG].apply(encode_trg)
test_df[SRC_LANG] = test_df[SRC_LANG].apply(encode_src)
test_df[TRG_LANG] = test_df[TRG_LANG].apply(encode_trg)

# Save new datasets
print("Saving datasets...")
train_df[SRC_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/train.tok.{SRC_LANG}", index=False, header=False)
train_df[TRG_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/train.tok.{TRG_LANG}", index=False, header=False)
dev_df[SRC_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/dev.tok.{SRC_LANG}", index=False, header=False)
dev_df[TRG_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/dev.tok.{TRG_LANG}", index=False, header=False)
test_df[SRC_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/test.tok.{SRC_LANG}", index=False, header=False)
test_df[TRG_LANG].to_csv(f"{DATASET_PATH}/preprocessed_fairseq/test.tok.{TRG_LANG}", index=False, header=False)

print("Preprocessing done!")

