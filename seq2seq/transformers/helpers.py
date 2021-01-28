import time
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets
from torchtext.data import Dataset

from tqdm import tqdm

from seq2seq import utils


def fit(model, train_iter, dev_iter, epochs, optimizer, criterion, checkpoint_path, clip=1.0,
        tr_writer=None, val_writer=None, tb_batch_rate=100):
    # Default values
    best_valid_loss = float('inf')

    # Fit model
    for epoch in range(epochs):
        start_time = time.time()
        n_iter = epoch + 1

        # Train model
        train_loss = train(model, train_iter, optimizer, criterion, clip,
                           n_iter=n_iter, tb_writer=tr_writer, tb_batch_rate=tb_batch_rate)

        # Evaluate model
        valid_loss = evaluate(model, dev_iter, criterion,
                              n_iter=n_iter, tb_writer=val_writer, tb_batch_rate=tb_batch_rate)

        # Checkpoint (in case there is an error later, do checkpoint first)
        diff_loss = valid_loss - best_valid_loss  # The lower, the better
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), checkpoint_path.format(epoch))
            best_valid_loss = valid_loss
            print("Checkpoint saved! Loss improvement: {:.5f}".format(diff_loss))

        # Summary report
        summary_report(train_loss, valid_loss, start_time=start_time,
                       tr_writer=tr_writer, val_writer=val_writer, n_iter=n_iter)



def train(model, train_iter, optimizer, criterion, clip, n_iter=None, tb_writer=None, tb_batch_rate=None):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        # Get data
        src, trg = batch.src, batch.trg

        # Reset grads and get output
        optimizer.zero_grad()

        ##############################
        # Feed input
        output, _ = model(src, trg[:, :-1])  # Ignore <eos> as input for trg

        # Reshape output / target
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and backward => CE(I(N, C), T(N))
        loss = criterion(output, trg)
        loss.backward()

        # Clip grads and update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Tensorboard
        if tb_writer and i % tb_batch_rate == 0:
            bn_iter = (n_iter-1) * len(train_iter) + (i+1)
            b_loss = epoch_loss / (i+1)
            tb_writer.add_scalar('Loss/batch', b_loss, bn_iter)
            tb_writer.add_scalar('PPL/batch', math.exp(b_loss), bn_iter)

    return epoch_loss / len(train_iter)


def evaluate(model, test_iter, criterion, n_iter=None, tb_writer=None, tb_batch_rate=None):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
            # Get data
            src, trg = batch.src, batch.trg

            ##############################
            # Feed input
            output, _ = model(src, trg[:, :-1])  # Ignore <eos> as input for trg

            # Reshape output / target
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
            trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
            ##############################

            # Compute loss and backward => CE(I(N, C), T(N))
            loss = criterion(output, trg)

            epoch_loss += loss.item()

            # Tensorboard
            if tb_writer and i % tb_batch_rate == 0:
                bn_iter = (n_iter - 1) * len(test_iter) + (i + 1)
                b_loss = epoch_loss / (i + 1)
                tb_writer.add_scalar('Loss/batch', b_loss, bn_iter)
                tb_writer.add_scalar('PPL/batch', math.exp(b_loss), bn_iter)

    return epoch_loss / len(test_iter)


def summary_report(train_loss=None, test_loss=None, start_time=None, tr_writer=None, val_writer=None, n_iter=0, testing=False):
    # Print summary
    if start_time:
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        print(f'Epoch: {n_iter:02} | Time: {epoch_mins}m {epoch_secs}s')
    else:
        print(f"Summary report:")

    # Metrics
    if train_loss is not None:
        # Metrics
        train_ppl = math.exp(train_loss)

        # Tensorboard
        if tr_writer:
            tr_writer.add_scalar('Loss', train_loss, n_iter)
            tr_writer.add_scalar('PPL', train_ppl, n_iter)

    # Validation
    if test_loss is not None:
        test_type = "Test" if testing else "Val."

        # Metrics
        test_ppl = math.exp(test_loss)
        print(f'\t {test_type} Loss: {test_loss:.3f} |  {test_type} PPL: {test_ppl:7.3f}')

        # Tensorboard
        if val_writer:
            val_writer.add_scalar('Loss', test_loss, n_iter)
            val_writer.add_scalar('PPL', test_ppl, n_iter)


