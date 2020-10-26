import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torchtext import data, datasets
from torchtext.data.metrics import bleu_score
import spacy
from tqdm import tqdm

import dill
from pathlib import Path
import time
import json
import math

import torch
from torchtext.data import Dataset


def train(model, train_iter, optimizer, criterion, clip, packed_pad=False, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        # Get data
        (src, src_len) = batch.src if packed_pad else (batch.src, None)
        trg = batch.trg

        # Reset grads and get output
        optimizer.zero_grad()

        # RNN
        if teacher_forcing_ratio is not None:
            output = model(src, src_len, trg, teacher_forcing_ratio) if packed_pad else model(src, trg, teacher_forcing_ratio)

            # Ignore <sos> token
            output, trg = output[1:], trg[1:]

            # Reshape output / target
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # (L, B, vocab) => (L*B, vocab)
            trg = trg.view(-1)  # (L, B) => (L*B) // We can use class numbers, no need for one-hot encoding

        else:  # Transformers
            output, _ = model(src, trg[:, :-1])  # Ignore <eos> as input for trg

            # Reshape output / target
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
            trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)

        # Compute loss and backward => CE(I(N, C), T(N))
        loss = criterion(output, trg)
        loss.backward()

        # Clip grads and update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(train_iter)


def evaluate(model, test_iter, criterion, packed_pad=False, teacher_forcing_ratio=0.0):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
            # Get data
            (src, src_len) = batch.src if packed_pad else (batch.src, None)
            trg = batch.trg

            # RNN
            if teacher_forcing_ratio is not None:
                # Get output (turn off teacher forcing)
                output = model(src, src_len, trg, teacher_forcing_ratio) if packed_pad else model(src, trg, teacher_forcing_ratio)

                # Ignore <sos> token
                output, trg = output[1:], trg[1:]

                # Reshape output / target
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)  # (L, B, vocab) => (L*B, vocab)
                trg = trg.view(-1)  # (L, B) => (L*B) // We can use class numbers, no need for one-hot encoding

            else:  # Transformers
                output, _ = model(src, trg[:, :-1])  # Ignore <eos> as input for trg

                # Reshape output / target
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
                trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)

            # Compute loss and backward => CE(I(N, C), T(N))
            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(test_iter)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("test.eps")
    print("save fig!")
    plt.close()


def calculate_bleu(model, data_iter, max_trg_len=150, packed_pad=False):
    trgs = []
    trg_pred = []

    model.eval()
    data_iter.batch_size = 1

    for batch in tqdm(data_iter, total=len(data_iter)):
        # Get data
        (src, src_len) = batch.src if packed_pad else (batch.src, None)
        trg = batch.trg

        # Get output
        if packed_pad:
            trg_indexes, _ = model.translate_sentence(src, src_len, max_trg_len)
        else:  # RNN, Transformers
            trg_indexes, _ = model.translate_sentence(src, max_trg_len)

        # Convert predicted indices to tokens
        trg_pred_tokens = [model.trg_field.vocab.itos[i] for i in trg_indexes]
        trg_tokens = [model.trg_field.vocab.itos[i] for i in trg.detach().cpu().int().flatten()]

        # Remove special tokens
        trg_pred_tokens = trg_pred_tokens[1:-1]  # Remove <sos> and <eos>
        trg_tokens = trg_tokens[1:-1]  # Remove <sos> and <eos>

        # Add predicted token
        trg_pred.append(trg_pred_tokens)
        trgs.append([trg_tokens])

    return bleu_score(trg_pred, trgs)


def save_dataset_examples(dataset, savepath):
    start = time.time()

    total = len(dataset.examples)
    with open(savepath, 'w') as f:
        # Save num. elements
        f.write(json.dumps(total))
        f.write("\n")

        # Save elements
        for pair in tqdm(dataset.examples, total=total):
            data = [pair.src, pair.trg]
            f.write(json.dumps(data))
            f.write("\n")

    end = time.time()
    print(f"Save dataset examples: [Total time= {end - start}; Num. examples={total}]")


def load_dataset(filename, fields):
    start = time.time()

    examples = []
    with open(filename, 'rb') as f:
        # Read num. elements
        line = f.readline()
        total = json.loads(line)

        # Load elements
        for i in tqdm(range(total), total=total):
            line = f.readline()
            example = json.loads(line)
            example = data.Example().fromlist(example, fields)  # Create Example obj.
            examples.append(example)

    # Build dataset ("examples" passed by reference)
    dataset = data.Dataset(examples, fields)

    end = time.time()
    print(f"Load dataset: [Total time= {end - start}; Num. examples={len(dataset.examples)}]")
    return dataset

