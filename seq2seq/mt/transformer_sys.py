import os
from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Strip, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing

from seq2seq.mt import transformer as tfmr


class LitTokenizer:
    def __init__(self, padding=False, truncation=False, max_length=None):
        super().__init__()
        self.SOS_WORD = '[SOS]'
        self.EOS_WORD = '[EOS]'
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.MASK_WORD = '[MASK]'
        self.special_tokens = [self.SOS_WORD, self.EOS_WORD, self.PAD_WORD, self.UNK_WORD, self.MASK_WORD]

        # Define tokenizers
        self.src_tokenizer, self.trg_tokenizer = self.configure_tokenizers(padding, truncation, max_length)


    def configure_tokenizers(self, padding, truncation, max_length):
        unk_idx = self.special_tokens.index(self.UNK_WORD)
        pad_idx = self.special_tokens.index(self.PAD_WORD)

        # Define template (common)
        basic_template = TemplateProcessing(
            single=f"{self.SOS_WORD} $A {self.EOS_WORD}",
            pair=f"{self.SOS_WORD} $A {self.EOS_WORD} {self.SOS_WORD} $B {self.EOS_WORD}",
            special_tokens=[(self.SOS_WORD, self.special_tokens.index(self.SOS_WORD)),
                            (self.EOS_WORD, self.special_tokens.index(self.EOS_WORD))],
        )

        # Settings
        pad_length = None
        if padding in {True, "longest"}:
            pass
        elif padding in {"max_length"}:
            pad_length = max_length
        elif padding in {False, "do_not_pad"}:
            pass
        else:
            raise ValueError("Unknown padding type")

        # SRC tokenizer
        src_tokenizer = Tokenizer(WordPiece())  # unk_token=... not working
        src_tokenizer.add_special_tokens(self.special_tokens)
        src_tokenizer.decoder = decoders.WordPiece()
        src_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), StripAccents()])  # StripAccents requires NFD
        src_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
        src_tokenizer.post_processor = basic_template
        if padding:
            src_tokenizer.enable_padding(pad_id=pad_idx, pad_token=self.PAD_WORD, length=pad_length)
        if truncation:
            src_tokenizer.enable_truncation(max_length, stride=0, strategy='longest_first')

        # TRG tokenizer
        trg_tokenizer = Tokenizer(WordPiece(unk_token=self.UNK_WORD))
        trg_tokenizer.add_special_tokens(self.special_tokens)
        trg_tokenizer.decoder = decoders.WordPiece()
        trg_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip()])
        trg_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
        trg_tokenizer.post_processor = basic_template
        if padding:
            trg_tokenizer.enable_padding(pad_id=pad_idx, pad_token=self.PAD_WORD, length=pad_length)
        if truncation:
            trg_tokenizer.enable_truncation(max_length, stride=0, strategy='longest_first')

        return src_tokenizer, trg_tokenizer

    def load_vocabs(self, src_vocab_file, trg_vocab_file):
        self.src_tokenizer.model = WordPiece.from_file(src_vocab_file, unk_token=self.UNK_WORD)
        self.trg_tokenizer.model = WordPiece.from_file(trg_vocab_file, unk_token=self.UNK_WORD)

    def train_vocabs(self, src_files, trg_files,
                     src_vocab_size=30000, trg_vocab_size=30000,
                     src_min_frequency=3, trg_min_frequency=3):
        # Define trainers
        src_trainer = WordPieceTrainer(vocab_size=src_vocab_size, min_frequency=src_min_frequency, special_tokens=self.special_tokens)
        trg_trainer = WordPieceTrainer(vocab_size=trg_vocab_size, min_frequency=trg_min_frequency, special_tokens=self.special_tokens)

        # Train tokenizers
        self.src_tokenizer.train(src_trainer, src_files)
        self.trg_tokenizer.train(trg_trainer, trg_files)

    def save_vocabs(self, folder, src_name="src", trg_name="trg"):
        # Source
        self.src_tokenizer.model.save(folder, src_name)
        self.src_tokenizer.save(f"{folder}/{src_name}-vocab.json")

        # Target
        self.trg_tokenizer.model.save(folder, trg_name)
        self.trg_tokenizer.save(f"{folder}/{trg_name}-vocab.json")

    def pad(self, examples, keys=None):
        pad_idx = self.special_tokens.index(self.PAD_WORD)

        # Keys to modify
        if not keys:
            keys = list(examples[0].keys())

        d = {}
        for k in keys:
            # Collect same-type items
            d[k] = [x[k] for x in examples]

            # Get max length
            max_length = max([x.shape[-1] for x in d[k]])

            # Apply padding
            for i, x in enumerate(examples):
                unpadded_t = x[k]
                if k == "ids":
                    tmp = torch.full((max_length,), fill_value=pad_idx, device=unpadded_t.device)  # All padding
                elif k == "attention_mask":
                    tmp = torch.full((max_length,), fill_value=0, device=unpadded_t.device)  # No attention mask
                else:
                    raise TypeError("Unknown key")
                tmp[:unpadded_t.shape[-1]] = unpadded_t
                d[k][i] = tmp
        return d


class LitTransfomer(pl.LightningModule):

    def __init__(self, tokenizer,
                 d_model=512,
                 enc_layers=2, dec_layers=2,
                 enc_heads=8, dec_heads=8,
                 enc_dff_dim=2048, dec_dff_dim=2048,
                 enc_dropout=0.1, dec_dropout=0.1,
                 max_src_len=2000, max_trg_len=2000,
                 batch_size=None,
                 learning_rate=10e-3):
        super().__init__()

        # Some variables
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Set tokenizer
        self.tokenizer = tokenizer

        # Vocab sizes
        input_dim = self.tokenizer.src_tokenizer.get_vocab_size()
        output_dim = self.tokenizer.trg_tokenizer.get_vocab_size()

        # Model
        self.enc = tfmr.Encoder(input_dim, d_model, enc_layers, enc_heads, enc_dff_dim, enc_dropout, max_src_len)
        self.dec = tfmr.Decoder(output_dim, d_model, dec_layers, dec_heads, dec_dff_dim, dec_dropout, max_trg_len)
        self.model = tfmr.Seq2Seq(self.enc, self.dec)

        # Initialize weights
        self.model.apply(tfmr.init_weights)

        # Set loss (ignore when the target token is <pad>)
        TRG_PAD_IDX = self.tokenizer.trg_tokenizer.token_to_id(self.tokenizer.PAD_WORD)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    def forward(self, x):
        # Inference
        return x

    def batch_step(self, batch, batch_idx):
        src, src_mask, trg, trg_mask = batch
        # trg_lengths = trg_mask.sum(dim=1) + 1  # Not needed

        # Feed input
        # src => whole sentence (including <sos>, <eos>)
        # src_mask => whole sentence (including <sos>, <eos>)
        # trg => whole sentence (except <eos> or use mask to remove it)
        # trg_mask => whole sentence except <eos>
        # NOTE: I remove the last token from TRG so that the prediction is L-1. This is
        # because later we will remove the <sos> from the TRG
        output, _ = self.model(src, src_mask, trg[:, :-1], trg_mask[:, :-1])

        # Reshape output / target
        # Let's presume that after the <eos> everything has be predicted as <pad>,
        # and then, we will ignore the pads in the CrossEntropy
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # (B, L, vocab) => (B*L, vocab)
        trg = trg[:, 1:].contiguous().view(-1)  # Remove <sos> and reshape to vector (B*L)
        ##############################

        # Compute loss and metrics
        losses = {'loss': self.criterion(output, trg)}
        metrics = {'ppl': math.exp(losses['loss'])}
        return losses, metrics

    def training_step(self, batch, batch_idx):
        # Run one mini-batch
        losses, metrics = self.batch_step(batch, batch_idx)

        # Logging to TensorBoard by default
        self.log('train_loss', losses['loss'])
        self.log('train_ppl', metrics['ppl'])
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        # Run one mini-batch
        losses, metrics = self.batch_step(batch, batch_idx)

        # Logging to TensorBoard by default
        self.log('val_loss', losses['loss'])
        self.log('val_ppl', metrics['ppl'])
        return losses['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
