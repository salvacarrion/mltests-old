import math
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(input_dim, output_dim, enc_emb_dim=256, dec_emb_dim=256,
               hid_dim=512, n_layers=1, enc_dropout=0.5, dec_dropout=0.5, device=None):
    # Set device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, dec_dropout)
    model = Seq2Seq(enc, dec, device).to(device)
    return model


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0.0, std=0.01)


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    """
    The decoder is going to receive one word at a time. Hence, it is gonna output
    one word after each forward
    """
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim+hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim*2+emb_dim, output_dim)

    def forward(self, trg, hidden, context):
        # Target => One word at a time
        embedded = self.dropout(self.embedding(trg.unsqueeze(0)))
        emb_cat = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(emb_cat, hidden)

        # Output => One word at a time
        output_cat = torch.cat([context, output, embedded], dim=2)
        prediction = self.out(output_cat.squeeze(0))

        return prediction, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, tf_ratio=0.5):
        # Vars
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Store outputs (L, B, 1)  => Indices
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # Run encoder
        hidden = self.encoder(src)
        context = hidden

        # first input to the decoder is the <sos> token
        dec_input = trg[0, :]  # Get first word index
        for t in range(1, max_len):
            output, hidden = self.decoder(dec_input, hidden, context)
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < tf_ratio
            top1 = output.max(1)[1]
            dec_input = (trg[t] if teacher_force else top1)

        return outputs
