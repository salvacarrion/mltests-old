import os
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets import load_dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from seq2seq.mt.transformer_sys import LitTokenizer, LitTransfomer

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    # Constants
    DATASET_PATH = f"../.data/miguel"
    SRC_LANG, TRG_LANG = ("en", "es")

    SRC_MIN_LENGTH = 1
    TRG_MIN_LENGTH = 1
    SRC_MAX_LENGTH = 150
    TRG_MAX_LENGTH = 150

    BATCH_SIZE = 64
    NUM_WORKERS = 12  # Problems with tokenizers
    PIN_MEMORY = True
    seed_everything(42)

    # Define Tokenizer
    # Do not use padding here. Datasets are preprocessed before batching
    tokenizer = LitTokenizer(padding=False, truncation=False)

    # Load vocab
    src_vocab_file = f"{DATASET_PATH}/vocab/{SRC_LANG}-vocab.txt"
    trg_vocab_file = f"{DATASET_PATH}/vocab/{TRG_LANG}-vocab.txt"
    tokenizer.load_vocabs(src_vocab_file, trg_vocab_file)

    # Get dataset (train/val)
    dataset = load_dataset('csv', data_files={'train': [f"{DATASET_PATH}/preprocessed/train.csv"],
                                              'validation': [f"{DATASET_PATH}/preprocessed/dev.csv"]})

    # Tokenize
    def encode(examples):
        # Encode strings
        _src_tokenized = tokenizer.src_tokenizer.encode_batch(examples[SRC_LANG])
        _trg_tokenized = tokenizer.trg_tokenizer.encode_batch(examples[TRG_LANG])

        # Select features
        src_tokenized = [{'ids': x.ids, 'attention_mask': x.attention_mask} for x in _src_tokenized]
        trg_tokenized = []
        for x in _trg_tokenized:
            mask = x.attention_mask
            mask[-1] = 0  # "Remove" <eos> for translation
            # lengths = len(x.attention_mask)  # needed due to padded inputs and masks
            trg_tokenized.append({'ids': x.ids, 'attention_mask': mask})  # , 'lengths': lengths
        new_examples = {'src': src_tokenized, 'trg': trg_tokenized}
        return new_examples

    def collate_fn(examples):
        # Decompose examples
        _src = [x['src'] for x in examples]
        _trg = [x['trg'] for x in examples]

        # Processed examples
        src = tokenizer.pad(_src, keys=['ids', 'attention_mask'])
        trg = tokenizer.pad(_trg, keys=['ids', 'attention_mask'])

        # Convert list to PyTorch tensor
        new_examples = (torch.stack(src['ids']), torch.stack(src['attention_mask']),
                        torch.stack(trg['ids']), torch.stack(trg['attention_mask']))
        return new_examples

    # Pre-process datasets (lazy)
    train_dataset = dataset['train'].map(encode, batched=True)
    val_dataset = dataset['validation'].map(encode, batched=True)

    # Filter entries with a length greater than max_length
    train_dataset = train_dataset.filter(lambda example: SRC_MIN_LENGTH <= len(example['src']['ids']) <= SRC_MAX_LENGTH and TRG_MIN_LENGTH <= len(example['trg']['ids']) <= TRG_MAX_LENGTH)
    val_dataset = val_dataset.filter(lambda example: SRC_MIN_LENGTH <= len(example['src']['ids']) <= SRC_MAX_LENGTH and TRG_MIN_LENGTH <= len(example['trg']['ids']) <= TRG_MAX_LENGTH)

    # Dataset formats
    train_dataset.set_format(type='torch', columns=['src', 'trg'])
    val_dataset.set_format(type='torch', columns=['src', 'trg'])

    # Dataset to Pytorch DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn)

    # init model
    model = LitTransfomer(tokenizer=tokenizer)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = Trainer.from_argparse_args(args)
    #trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
    print("Done!")
