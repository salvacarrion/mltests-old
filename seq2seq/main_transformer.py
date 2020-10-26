import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets
import pandas as pd
from tqdm import tqdm

import spacy
import numpy as np

import random
import math
import time

from seq2seq import utils, helpers
import re
import os
import pickle

# Build model and initialize
MODEL_NAME = "s2s_6_transformer"
DATASET_NAME = "miguel"  # multi30k, miguel
DATASET_PATH = f".data/{DATASET_NAME}"
TRAIN = True
EVALUATE = True
BLUE = True
LEARNING_RATE = 0.0005
MIN_FREQ = 5
MAX_SIZE = 8000 - 4  # 4 reserved words <sos>, <eos>, <pad>, <unk>
N_EPOCHS = 2048
MAX_SRC_LENGTH = 100
MAX_TRG_LENGTH = 100
BATCH_SIZE = 64

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up fields
SOS_WORD = '<sos>'
EOS_WORD = '<eos>'

# Set fields
SRC = data.Field(tokenize='spacy', tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
TRG = data.Field(tokenize='spacy', tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
fields = [('src', SRC), ('trg', TRG)]

# Load examples
train_data = helpers.load_dataset_examples(f"{DATASET_PATH}/tokenized/train.json")
dev_data = helpers.load_dataset_examples(f"{DATASET_PATH}/tokenized/dev.json")
test_data = helpers.load_dataset_examples(f"{DATASET_PATH}/tokenized/test.json")

# (Re)Build dataset (fast)
train_data = data.Dataset(train_data, fields)
dev_data = data.Dataset(dev_data, fields)
test_data = data.Dataset(test_data, fields)

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(dev_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

SRC.build_vocab(train_data.src, max_size=MAX_SIZE)
TRG.build_vocab(train_data.trg, max_size=MAX_SIZE)

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (es) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train_data, dev_data, test_data), batch_size=BATCH_SIZE, device=device,
    sort=False
)


if MODEL_NAME == "s2s_6_transformer":
    from seq2seq.models import s2s_6_transfomer as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG, max_src_len=MAX_SRC_LENGTH, max_trg_len=MAX_TRG_LENGTH)
    model.apply(s2s_model.init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

else:
    raise ValueError("Unknown model name")

print(f"Selected model: {MODEL_NAME}")
print(f'The model has {utils.count_parameters(model):,} trainable parameters')

# Set loss (ignore when the target token is <pad>)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Check if it is using the PGU
print(utils.gpu_info())

# Train and validate model
CLIP = 1.0
best_valid_loss = float('inf')
checkpoint_path = f'checkpoints/checkpoint_{MODEL_NAME}.pt'
packed_pad = False
teacher_forcing_ratio = None

if TRAIN:
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train model
        train_loss = helpers.train(model, train_iter, optimizer, criterion, CLIP, packed_pad, teacher_forcing_ratio)

        # Evaluate model
        valid_loss = helpers.evaluate(model, dev_iter, criterion, packed_pad, teacher_forcing_ratio)

        # Print epoch results
        end_time = time.time()
        epoch_mins, epoch_secs = helpers.epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved!")

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Testing
if EVALUATE or BLUE:
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print("Model loaded")

    # Evaluate best model
    if EVALUATE:
        test_loss = helpers.evaluate(model, test_iter, criterion, packed_pad, teacher_forcing_ratio)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    if BLUE:
        bleu_score = helpers.calculate_bleu(model, test_iter, max_trg_len=MAX_TRG_LENGTH, packed_pad=packed_pad)
        print(f'BLEU score = {bleu_score * 100:.2f}')


# # Translate sente
# example_idx = 12
#
# src = vars(train_data.examples[example_idx])['src']
# trg = vars(train_data.examples[example_idx])['trg']
# print(f'src = {" ".join(src)}')
# print(f'trg = {" ".join(trg)}')
#
# translation, attention = translate_sentence(src, SRC, TRG, model, device)
# print(f'predicted trg = {" ".join(translation)}')
#
# # Display attention
# display_attention(src, translation, attention)


print("Done!")
