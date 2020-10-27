import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


def make_model(src_field, trg_field, enc_emb_dim=256, dec_emb_dim=256,
               enc_hid_dim=512, dec_hid_dim=512, enc_dropout=0.5, dec_dropout=0.5,
               device=None, data_parallelism=False):
    # Input/Output dims
    input_dim = len(src_field.vocab)
    output_dim = len(trg_field.vocab)

    # Build model
    attn = BahdanauAttention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)
    model = Seq2Seq(enc, dec, src_field, trg_field, device).to(device)
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    # Allow parallelization
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

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch
        packed_outputs, hidden = self.rnn(packed_embedded)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # Concat hiding layers across "embedding"
        h_fw = hidden[-2, :, :]
        h_bw = hidden[-1, :, :]
        hidden = torch.cat([h_fw, h_bw], dim=1)

        # Transform the double context vector into one.
        # This is done because the decoder is not bidirectional
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden

class BahdanauAttention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.fc_hidden = nn.Linear(self.dec_hid_dim, self.dec_hid_dim, bias=False)
        self.fc_encoder = nn.Linear(self.dec_hid_dim*2, self.dec_hid_dim, bias=False)
        self.v = nn.Linear(self.dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # Calculate alignment scores
        decoder_hidden = decoder_hidden.unsqueeze(0)

        # Multi-dimension linears are apply to the last dimension
        score1 = self.fc_hidden(decoder_hidden)  # (1, B, D) => # (1, B, D)
        score2 = self.fc_encoder(encoder_outputs)  # (L, B, E) => (L, B, D)

        # Compute scores
        alignment_scores = torch.tanh(score1+score2)  # score1 is broadcasted to score2
        alignment_scores = self.v(alignment_scores)  # (L, B, D) [x (1, D, 1)] => (L, B, 1)
        alignment_scores = alignment_scores.squeeze(2)  # (L, B, 1) => (L, B)
        alignment_scores = alignment_scores.T  # (L, B) => (B, L)

        # Put -inf where there is padding (The softmax will make them zeros)
        alignment_scores = alignment_scores.masked_fill(mask == 0, -np.inf)

        # Softmax alignment scores
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

    def forward(self, trg, decoder_hidden, encoder_outputs, mask):
        # Target => One word at a time
        embedded = self.dropout(self.embedding(trg.unsqueeze(0)))

        # Compute attention
        attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)

        # Reshape
        attn_weights = attn_weights.unsqueeze(1)  # (B, L) => (B, 1, L)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (L, B, E) => (B, L, E)

        # Context vector: Encoder * Softmax(alignment scores)
        context_vector = torch.bmm(attn_weights, encoder_outputs)  # (B, 1, L) x (B, L, E) => (B, 1, E)
        context_vector = context_vector.permute(1, 0, 2)  # (B, 1, E) => (1, B, E)  // 1 == "L"

        # Concatenate embedding word and context vector
        cat_input = torch.cat([embedded, context_vector], dim=2)  # (1, B, E1), (1, B, E2) => (1, B, E1+E2)
        output, decoder_hidden = self.rnn(cat_input, decoder_hidden.unsqueeze(0))

        prediction = self.out(output.squeeze(0))  # (B, D) => (B, O)
        return prediction, decoder_hidden.squeeze(0), attn_weights.squeeze(1)


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, src_field, trg_field, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_field = src_field
        self.trg_field = trg_field
        self.device = device

    def create_mask(self, src):
        pad_idx = self.src_field.vocab.stoi[self.src_field.pad_token]
        mask = (src != pad_idx)
        mask = mask.T  # [L, B] => [B, L]
        return mask

    def forward(self, src, src_len, trg, tf_ratio=0.5):
        # Vars
        batch_size = trg.shape[1]
        trg_max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Store outputs (L, B, 1)  => Indices
        outputs = torch.zeros(trg_max_len, batch_size, trg_vocab_size).to(self.device)

        # Run encoder
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> token
        dec_input = trg[0, :]  # Get first word index
        # There is no point in setting output[0] to <sos> since it will be ignored later

        # Get masks for src != <pad> => [0 0 0 0 0 1 1]
        mask = self.create_mask(src)

        # Iterate over target (max) length
        for t in range(1, trg_max_len):
            output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, mask)
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < tf_ratio
            if teacher_force:  # Use actual token
                dec_input = trg[t]
            else:
                top1 = output.max(1)[1]  # [0]=>values; [1]=>indices
                dec_input = top1

        return outputs

    def translate_sentence(self, src, src_len, max_trg_len=50):
        # Single sentence ("unlimited length")

        # Get special indices
        sos_idx = self.trg_field.vocab.stoi[self.trg_field.init_token]
        eos_idx = self.trg_field.vocab.stoi[self.trg_field.eos_token]

        # Run encoder
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_len)

        # Create mask
        mask = self.create_mask(src)

        # Initialize attention matrix
        attentions = torch.zeros(max_trg_len, 1, len(src)).to(self.device)

        # Set fist word (<sos>)
        trg_indexes = [sos_idx]

        for i in range(max_trg_len):

            # Get last predicted word
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, attention = self.decoder(trg_tensor, hidden, encoder_outputs, mask)

            # Save attention
            attentions[i] = attention  # (i, 1, Lsrc) <= (1, Lsrc)

            # Get predicted token
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            # If predicted token == <eos> => stop
            if pred_token == eos_idx:
                break

        return trg_indexes, attentions