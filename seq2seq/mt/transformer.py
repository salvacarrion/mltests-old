import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils, search


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)  # Vocab => emb
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # Pos => emb_pos

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        assert src_len <= self.max_length

        # Initial positions: 0,1,2,... for each sample
        device = src.device
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)  # (B, src_len)

        # Mix token embeddings and positional embeddings
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))  # (B, src_len, hid_dim)

        for layer in self.layers:
            src = layer(src, src_mask)  # (B, src_len, hid_dim)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Multi-head attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # Feedforward
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Self-attention (query = key = value = x)
        Q = self.fc_q(query)  # (B, L, hid_dim) => (B, L, hid_dim)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # From (B, len, dim) => (B, len, n_heads, head_dim) => (B, n_heads, len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Get un-normalized attention.
        # Transpose Keys => (q_len, head_dim) x (head_dim, k_len) = (q_len, k_len)
        K_t = K.permute(0, 1, 3, 2)
        energy = torch.matmul(Q, K_t) / self.scale

        # Ignore pads
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Normalize attention
        attention = torch.softmax(energy, dim=-1)

        # Encode input with attention (k_len == v_len)
        x = torch.matmul(self.dropout(attention), V)  # [..., q_len, k_len] x [..., v_len, head dim]

        # Go back to the input size
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, n_heads, len, head_dim) => (B, len, n_heads, head_dim)
        x = x.view(batch_size, -1, self.hid_dim)  # (B, len, n_heads, head_dim) => (B, len, hid_dim)

        # Linear
        x = self.fc_o(x)  # (..., hid_dim) => (..., hid_dim)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))  # Expand
        x = self.fc_2(x)  # Compress
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()

        self.max_length = max_length

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # This limits decoding length at testing

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        assert trg_len <= self.max_length

        # Initial positions: 0,1,2,... for each sample
        device = trg.device
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # Mix token embeddings and positional embeddings
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,  pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Self-attention (target + mask)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # Encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # Position-wise feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.softmax = nn.Softmax(dim=2)

    def make_src_mask(self, src_mask):
        # Extend dimensions
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)
        return src_mask

    def make_trg_mask(self, src_mask):
        # Extend dimensions
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)

        # Diagonal matrix to hide next token (LxL)
        trg_len = src_mask.shape[1]  # target (max) length
        trg_tri_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()

        # Add pads to the diagonal matrix (LxL)&Pad
        # This is automatically broadcasted (B, 1, 1, L) & (L, L) => (B, 1, L, L)
        trg_mask = src_mask & trg_tri_mask
        return trg_mask

    def forward(self, src, src_mask, trg, trg_mask):
        # Process masks
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask)

        # Encoder-Decoder
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention
