import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets
import pandas as pd

import spacy
import numpy as np

import random
import math
import time

from seq2seq import utils, helpers

# Build model and initialize
MODEL_NAME = "s2s_6_transformer"
DATASET_NAME = "miguel"  # multi30k
DATASET_PATH = f".data/{DATASET_NAME}"
TRAIN = False
EVALUATE = True
BLUE = True
LEARNING_RATE = 0.0005

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

    # Convert to pandas
    data_raw = {lang_src: [line for line in file_src], lang_trg: [line for line in file_trg]}
    data_df = pd.DataFrame(data_raw, columns=[lang_src, lang_trg])

    return data_df


if DATASET_NAME == "miguel":
    # # Training
    train_df = clean_file(file_src=f'{DATASET_PATH}/europarl.en', file_trg=f'{DATASET_PATH}/europarl.es', lang_src='English', lang_trg='Spanish')
    train_df.to_csv(f"{DATASET_PATH}/clean/train.csv", index=False)

    # Validation
    val_df = clean_file(file_src=f'{DATASET_PATH}/dev.en', file_trg=f'{DATASET_PATH}/dev.es', lang_src='English', lang_trg='Spanish')
    val_df.to_csv(f"{DATASET_PATH}/clean/val.csv", index=False)

    # Test
    test_df = clean_file(file_src=f'{DATASET_PATH}/test.en', file_trg=f'{DATASET_PATH}/test.es', lang_src='English', lang_trg='Spanish')
    test_df.to_csv(f"{DATASET_PATH}/clean/test.csv", index=False)

    SRC = data.Field(tokenize="spacy", tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
    TRG = data.Field(tokenize="spacy", tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)

    # Tokenize dataset
    train_data, val_data, test_data = data.TabularDataset.splits(path=f'{DATASET_PATH}/clean/',
                                                                 train='train.csv', validation='val.csv', test='test.csv',
                                                                 format='csv', fields=(SRC, TRG))

    asdas = 33

if DATASET_NAME == "multi30k":
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

# Create vocabulary
MIN_FREQ = 2
SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)

print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 128
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=BATCH_SIZE, device=device,
)


if MODEL_NAME == "s2s_6_transformer":
    from seq2seq.models import s2s_6_transfomer as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG)
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
N_EPOCHS = 10
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
        bleu_score = helpers.calculate_bleu(model, test_iter, packed_pad=packed_pad)
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
