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
TRAIN = False
EVALUATE = False
BLUE = True
LEARNING_RATE = 0.0005
MIN_FREQ = 5
MAX_SIZE = 10000

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


def clean_file(file_src, file_trg, lang_src, lang_trg):
    file_src = open(file_src, encoding='utf-8').read().split('\n')
    file_trg = open(file_trg, encoding='utf-8').read().split('\n')
    assert len(file_src) == len(file_trg)

    # Preprocess lines
    p = re.compile("^<seg id=\"\d+\">")
    for i in tqdm(range(len(file_src)), total=len(file_src)):
        # Remove html
        file_src[i] = p.sub('', file_src[i])
        file_trg[i] = p.sub('', file_trg[i])

    # Convert to pandas
    data_raw = {lang_src: [line for line in file_src], lang_trg: [line for line in file_trg]}
    data_df = pd.DataFrame(data_raw, columns=[lang_src, lang_trg])

    return data_df


SRC = data.Field(tokenize='spacy', tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
TRG = data.Field(tokenize='spacy', tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)

if DATASET_NAME == "miguel":
    # Check clean directory
    CLEAN_PATH = f"{DATASET_PATH}/clean"
    if not os.path.exists(CLEAN_PATH):
        print(f"Clean dataset directory does not exists: {CLEAN_PATH}")

    # Training
    train_fname = f"{CLEAN_PATH}/train.csv"
    if not os.path.exists(train_fname):
        print(f"- Parsing training...")
        train_df = clean_file(file_src=f'{DATASET_PATH}/europarl.en', file_trg=f'{DATASET_PATH}/europarl.es', lang_src='English', lang_trg='Spanish')
        train_df.to_csv(train_fname, index=False)
        print(f"- Train dataset saved! ({train_fname})")

    # Validation
    val_fname = f"{CLEAN_PATH}/val.csv"
    if not os.path.exists(val_fname):
        print(f"- Parsing validation...")
        val_df = clean_file(file_src=f'{DATASET_PATH}/dev.en', file_trg=f'{DATASET_PATH}/dev.es', lang_src='English', lang_trg='Spanish')
        val_df.to_csv(val_fname, index=False)
        print(f"- Validation dataset saved! ({val_fname})")

    # Test
    test_fname = f"{CLEAN_PATH}/test.csv"
    if not os.path.exists(test_fname):
        print(f"- Parsing testing...")
        test_df = clean_file(file_src=f'{DATASET_PATH}/test.en', file_trg=f'{DATASET_PATH}/test.es', lang_src='English', lang_trg='Spanish')
        test_df.to_csv(test_fname, index=False)
        print(f"- Test dataset saved! ({test_fname})")


    # Tokenize dataset
    clean_ds = f"{DATASET_PATH}/clean_ds"
    if not os.path.exists(clean_ds):
        print("Reading and preprocessing Tabular dataset...")
        data_fields = [('src', SRC), ('trg', TRG)]
        train_data, val_data, test_data = data.TabularDataset.splits(path=f'{DATASET_PATH}/clean/',
                                                                     train='train.csv', validation='val.csv', test='test.csv',
                                                                     format='csv', fields=data_fields, skip_header=True)
        # Save preprocessed
        print("Saving dataset...")
        # helpers.save_dataset(train_data, f"{DATASET_PATH}/clean_ds", "train_data")
        helpers.save_dataset(train_data, f"{DATASET_PATH}/clean_ds", "train_data")
        helpers.save_dataset(val_data, f"{DATASET_PATH}/clean_ds", "val_data")
        helpers.save_dataset(test_data, f"{DATASET_PATH}/clean_ds", "test_data")
        print("Datasets saved!")

    else:
        print("Loading datasets...")

        train_data = helpers.load_dataset(f"{DATASET_PATH}/clean_ds", "test_data")
        val_data = helpers.load_dataset(f"{DATASET_PATH}/clean_ds", "val_data")
        test_data = helpers.load_dataset(f"{DATASET_PATH}/clean_ds", "test_data")
        print("Datasets loaded!")

        # Fix references
        SRC = train_data.fields['src']
        TRG = train_data.fields['trg']
        val_data.fields['src'] = SRC
        val_data.fields['trg'] = TRG
        test_data.fields['src'] = SRC
        test_data.fields['trg'] = TRG

elif DATASET_NAME == "multi30k":
    SRC = data.Field(tokenize="spacy", tokenizer_language="de", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
    TRG = data.Field(tokenize="spacy", tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)

    # Download and tokenize dataset
    train_data, val_data, test_data = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))


else:
    raise ValueError("Unknown dataset")

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(val_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

# Get sentence lengths
sent_lenghts = np.array([(len(sent.src), len(sent.trg)) for sent in train_data.examples])
min_lengths = np.min(sent_lenghts, axis=0)
max_lengths = np.max(sent_lenghts, axis=0)

print(f"Length range for SRC: {min_lengths[0]}-{max_lengths[0]}")
print(f"Length range for TRG: {min_lengths[1]}-{max_lengths[1]}")

# Get sentence lengths
sent_lenghts = np.array([(len(sent.src), len(sent.trg)) for sent in val_data.examples])
min_lengths = np.min(sent_lenghts, axis=0)
max_lengths = np.max(sent_lenghts, axis=0)

print(f"Length range for SRC (val): {min_lengths[0]}-{max_lengths[0]}")
print(f"Length range for TRG (val): {min_lengths[1]}-{max_lengths[1]}")

# Get sentence lengths
sent_lenghts = np.array([(len(sent.src), len(sent.trg)) for sent in test_data.examples])
min_lengths = np.min(sent_lenghts, axis=0)
max_lengths = np.max(sent_lenghts, axis=0)

print(f"Length range for SRC (test): {min_lengths[0]}-{max_lengths[0]}")
print(f"Length range for TRG (test): {min_lengths[1]}-{max_lengths[1]}")


SRC.build_vocab(train_data.src, max_size=MAX_SIZE)
TRG.build_vocab(train_data.trg, max_size=MAX_SIZE)

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (es) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=BATCH_SIZE, device=device,
    sort=False
)


if MODEL_NAME == "s2s_6_transformer":
    from seq2seq.models import s2s_6_transfomer as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG, max_src_len=845, max_trg_len=760)
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
N_EPOCHS = 5
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
        valid_loss = helpers.evaluate(model, val_iter, criterion, packed_pad, teacher_forcing_ratio)

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
        bleu_score = helpers.calculate_bleu(model, test_iter, max_trg_len=760, packed_pad=packed_pad)
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
