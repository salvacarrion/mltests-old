import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils, search


def make_model(src_field, trg_field, hid_dim=256, enc_layers=3, dec_layers=3, enc_heads=8, dec_heads=8,
               enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.1, dec_dropout=0.1,
               max_src_len=100, max_trg_len=100, device=None, data_parallelism=False):

    # Input/Output dims
    input_dim = len(src_field.vocab)
    output_dim = len(trg_field.vocab)

    # Required to compute all heads with a single matrix multiplication
    assert hid_dim % enc_heads == 0
    assert hid_dim % dec_heads == 0

    # Build model
    enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, max_src_len, device)
    dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, max_trg_len, device)
    model = Seq2Seq(enc, dec, src_field, trg_field, device)
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    # Parallelize model
    device_count = torch.cuda.device_count()
    device_ids = list(range(torch.cuda.device_count()))
    print(f"Data parallelism: {data_parallelism}")
    if data_parallelism and device_count > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"\t- Num. devices: {torch.cuda.device_count()}")
        print(f"\t- Device IDs:{str(device_ids)}")

    # Send to device
    model.to(device)
    return model


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length, device):
        super().__init__()
        self.device = device
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)  # Vocab => emb
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # Pos => emb_pos

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        assert src_len <= self.max_length

        # Initial positions: 0,1,2,... for each sample
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # (B, src_len)

        # Mix token embeddings and positional embeddings
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))  # (B, src_len, hid_dim)

        for layer in self.layers:
            src = layer(src, src_mask)  # (B, src_len, hid_dim)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
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
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.device = device
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
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length, device):
        super().__init__()

        self.device = device
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # This limits decoding length at testing

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        assert trg_len <= self.max_length

        # Initial positions: 0,1,2,... for each sample
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # Mix token embeddings and positional embeddings
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

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

    def __init__(self, encoder, decoder, src_field, trg_field, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_field = src_field
        self.trg_field = trg_field
        self.device = device

        self.softmax = nn.Softmax(dim=2)

    def make_src_mask(self, src):
        # Mask <pad>
        pad_idx = self.src_field.vocab.stoi[self.src_field.pad_token]
        src_mask = (src != pad_idx)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)
        return src_mask

    def make_trg_mask(self, trg):

        # Mask <pad>
        pad_idx = self.trg_field.vocab.stoi[self.trg_field.pad_token]
        trg_pad_mask = (trg != pad_idx)
        trg_pad_mask = trg_pad_mask.unsqueeze(1).unsqueeze(2)  #  (B, n_heads=1, seq_len=1, seq_len)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        # Diagonal matrix to hide next token (LxL)
        trg_len = trg.shape[1]  # target (max) length
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]

        # Add pads to the diagonal matrix (LxL)&Pad
        # This is automatically broadcasted (B, 1, 1, L) & (L, L) => (B, 1, L, L)
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        src_max_idx = src.max()
        trg_max_idx = src.max()

        assert src_max_idx < len(self.src_field.vocab)
        assert trg_max_idx < len(self.trg_field.vocab)

        # Build masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # Encoder-Decoder
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention

    def decode_word(self, enc_src, src_mask, trg_indexes):
            # Get predicted words (all)
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)  # (1, 1->L)

            # Build target mask
            trg_mask = self.make_trg_mask(trg_tensor)  # (B, n_heads, L, L)

            with torch.no_grad():
                # Inputs: source + current translation
                output, last_attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)  # (B, L, output_dim)

            # Find top k words from the output vocabulary
            output = self.softmax(output)
            probs, idxs = output.sort(descending=True)

            # Get last word
            probs, idxs = probs[:, -1, :], idxs[:, -1, :]
            return (probs, idxs), last_attention

    def translate_sentence(self, src, max_trg_len, beam_width):
        # Single sentence ("unlimited length")

        # Get special indices
        sos_idx = self.trg_field.vocab.stoi[self.trg_field.init_token]
        eos_idx = self.trg_field.vocab.stoi[self.trg_field.eos_token]

        # Build source mask
        src_mask = self.make_src_mask(src)

        # Run encoder
        with torch.no_grad():
            enc_src = self.encoder(src, src_mask)

        # Set fist word (<sos>)
        trg_indexes = [([sos_idx], [1.0], None)]  # Common starting point (indices, probs, attn)
        for t in range(max_trg_len):
            # Get top-k next token. None if ended
            top_k_beam, last_attention = search.get_top_tokens(self, (enc_src, src_mask), trg_indexes, beam_width, eos_idx, max_trg_len)

            if all([x is None for x in top_k_beam]):
                break
            else:
                if beam_width == 1:
                    new_top_k = search.greedy_search(trg_indexes, top_k_beam, last_attention)
                else:
                    new_top_k = search.beam_search(trg_indexes, top_k_beam, beam_width, last_attention)
                new_trg_indexes = search.update_top_k(trg_indexes, new_top_k, beam_width, eos_idx)
                trg_indexes = new_trg_indexes

        return trg_indexes

    def translate(self, sentence, max_length, beam_width):
        # Process sentence
        src_preprocessed = self.src_field.preprocess(sentence)
        src_indexes = self.src_field.process([src_preprocessed], self.device)

        trg_pack = self.translate_sentence(src_indexes, max_trg_len=max_length, beam_width=beam_width)

        # Convert to CPU
        src_indexes = src_indexes.cpu().int().flatten()

        # Convert predicted indices to tokens
        src_tokens = [self.src_field.vocab.itos[int(i)] for i in src_indexes]
        trg_pred_tokens, attns = [], []
        for trg in trg_pack:
            trg_pred_tokens.append([self.trg_field.vocab.itos[int(i)] for i in trg[0]])
            attns.append(trg[2].cpu() if trg[2] is not None else None)  # Get attention

        # Minor clean
        trg_pred_tokens = [x[1:] for x in trg_pred_tokens]  # Remove <sos> since it's given
        return (src_tokens, trg_pred_tokens), attns
