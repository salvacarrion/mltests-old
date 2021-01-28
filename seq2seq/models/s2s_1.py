import math
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(input_dim, output_dim, enc_emb_dim=256, dec_emb_dim=256,
               hid_dim=512, n_layers=2, enc_dropout=0.5, dec_dropout=0.5, device=None):
    # Set device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)
    model = Seq2Seq(enc, dec, device).to(device)
    return model


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """
    The decoder is going to receive one word at a time. Hence, it is gonna output
    one word after each forward
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, trg, hidden, cell):
        # Target => One word at a time
        embedded = self.dropout(self.embedding(trg.unsqueeze(0)))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Output => One word at a time
        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, tf_ratio=0.5):
        # Vars
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Store outputs (L, B, 1)  => Indices
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # Run encoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token
        dec_input = trg[0, :]  # Get first word index
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < tf_ratio
            top1 = output.max(1)[1]
            dec_input = (trg[t] if teacher_force else top1)

        return outputs
