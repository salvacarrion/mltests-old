import math
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(input_dim, output_dim, enc_emb_dim=256, dec_emb_dim=256,
               enc_hid_dim=512, dec_hid_dim=512, attn_dim=64, n_layers=1,
               enc_dropout=0.5, dec_dropout=0.5, device=None):
    # Set device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    attn = BahdanauAttention(enc_hid_dim, dec_hid_dim, attn_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    return model


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Concat hiding layers across "embedding"
        h_fw = hidden[-2, :, :]
        h_bw = hidden[-1, :, :]
        hidden = torch.cat([h_fw, h_bw], dim=1)

        # Transform the double context vector into one.
        # This is done because the decoder is not bidirectional
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden


class BahdanauAttention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.fc_hidden = nn.Linear(self.dec_hid_dim, self.dec_hid_dim, bias=False)
        self.fc_encoder = nn.Linear(self.dec_hid_dim*2, self.dec_hid_dim, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, self.dec_hid_dim))

    def forward(self, decoder_hidden, encoder_outputs):
        # Calculate alignment scores
        decoder_hidden = decoder_hidden.unsqueeze(0)
        alignment_scores = torch.tanh(self.fc_hidden(decoder_hidden) + self.fc_encoder(encoder_outputs))
        alignment_scores = torch.matmul(alignment_scores, self.weight.unsqueeze(2))

        # Softmax alignment scores
        alignment_scores = alignment_scores.permute(1, 0, 2)
        attn_weights = F.softmax(alignment_scores, dim=1)
        return attn_weights


class Decoder(nn.Module):
    """
    The decoder is going to receive one word at a time. Hence, it is gonna output
    one word after each forward
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(dec_hid_dim, output_dim)

    def forward(self, trg, decoder_hidden, encoder_outputs):
        # Target => One word at a time
        embedded = self.dropout(self.embedding(trg.unsqueeze(0)))

        # Compute attention
        attn_weights = self.attention(decoder_hidden, encoder_outputs)

        # Context vector: Encoder * Softmax(alignment scores)
        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), encoder_outputs.permute(1, 0, 2))
        context_vector = context_vector.permute(1, 0, 2)

        # Concatenate embedding word and context vector
        cat_input = torch.cat([embedded, context_vector], dim=2)
        output, decoder_hidden = self.rnn(cat_input, decoder_hidden.unsqueeze(0))

        prediction = self.out(output.squeeze(0))

        return prediction, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Vars
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Store outputs (L, B, 1)  => Indices
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # Run encoder
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        dec_input = trg[0, :]  # Get first word index
        for t in range(1, max_len):
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]  # [0]=>values; [1]=>indices
            dec_input = (trg[t] if teacher_force else top1)

        return outputs
