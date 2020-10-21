import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets

import spacy
import numpy as np

import random
import math
import time

from seq2seq import utils, helpers

# Build model and initialize
# "s2s_4_bahdanau", "s2s_4_ba_lu_mixed", "s2s_4_luong"
MODEL_NAME = "s2s_4_luong"
TRAIN = True
EVALUATE = True
BLUE = True

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
SRC = data.Field(tokenize="spacy", tokenizer_language="de", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, include_lengths=True)
TRG = data.Field(tokenize="spacy", tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, include_lengths=False)

# Download and tokenize dataset
train_data, val_data, test_data = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

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


# Packed padded sequences need to be sorted by their non-padded lengths in descending order
def sort_batch(x):
    return len(x.src)


BATCH_SIZE = 128
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=BATCH_SIZE, device=device,
    sort_within_batch=True,
    sort_key=sort_batch,
)


if MODEL_NAME == "s2s_4_bahdanau":
    from seq2seq.models import s2s_4_bahdanau as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG)
    model.apply(s2s_model.init_weights)
    optimizer = optim.Adam(model.parameters())

elif MODEL_NAME == "s2s_4_ba_lu_mixed":
    from seq2seq.models import s2s_4_ba_lu_mixed as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG)
    model.apply(s2s_model.init_weights)
    optimizer = optim.Adam(model.parameters())

elif MODEL_NAME == "s2s_4_luong":
    from seq2seq.models import s2s_4_luong as s2s_model

    model = s2s_model.make_model(src_field=SRC, trg_field=TRG)
    model.apply(s2s_model.init_weights)
    optimizer = optim.Adam(model.parameters())

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
packed_pad = True

if TRAIN:
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train model
        train_loss = helpers.train(model, train_iter, optimizer, criterion, CLIP, packed_pad)

        # Evaluate model
        valid_loss = helpers.evaluate(model, val_iter, criterion, packed_pad)

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
        test_loss = helpers.evaluate(model, test_iter, criterion, packed_pad)
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
