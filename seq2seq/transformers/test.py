

import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torch.utils.tensorboard import SummaryWriter

from seq2seq.transformers import helpers
from seq2seq import utils

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import seaborn as sns
# sns.set() Problems with attention

import re

import spacy
nlp_en = spacy.load("en")
nlp_es = spacy.load("es")


# Names
EXPERIMENT_NAME = "runs/transformer"
MODEL_NAME = "simple_transformer"

# Build model and initialize
DATASET_NAME = "miguel"  # multi30k, miguel
DATASET_PATH = f"../.data/{DATASET_NAME}"
ALLOW_DATA_PARALLELISM = False
MAX_SRC_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
MAX_TRG_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
MAX_TRG_LENGTH_TEST = int(MAX_TRG_LENGTH * 1.0)  # len>1.0 is not supported by all models
CHECKPOINT_PATH = f'checkpoints/simple_transformer_2.pt'
SOS_WORD = '<sos>'
EOS_WORD = '<eos>'

# Set fields
SRC = data.Field(tokenize='spacy', tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
TRG = data.Field(tokenize='spacy', tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
fields = [('src', SRC), ('trg', TRG)]

# Load vocabulary
src_vocab = utils.load_vocabulary(f'{DATASET_PATH}/tokenized/src_vocab.pkl')
trg_vocab = utils.load_vocabulary(f'{DATASET_PATH}/tokenized/trg_vocab.pkl')
print("Vocabularies loaded!")

# Add vocabularies to fields
SRC.vocab = src_vocab
TRG.vocab = trg_vocab
print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (es) vocabulary: {len(TRG.vocab)}")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(utils.gpu_info())


from seq2seq.models import s2s_6_transfomer as builder
model = builder.make_model(src_field=SRC, trg_field=TRG,
                           max_src_len=MAX_SRC_LENGTH, max_trg_len=MAX_TRG_LENGTH, device=device,
                           data_parallelism=ALLOW_DATA_PARALLELISM)

# Load best model
model.load_state_dict(torch.load(CHECKPOINT_PATH))
print("Model loaded!")

# Set for evaluation
model.eval()

# Translate
src_sentence = "Barack Hussein Obama II is an American politician and attorney who served as the 44th president of the United States of America from 2009 to 2017."

# Beam 1
(src_tokens, trans_tokens), attns1 = model.translate(src_sentence, max_length=MAX_TRG_LENGTH_TEST, beam_width=1)
utils.show_translation_pair(src_tokens, trans_tokens, nlp_src=nlp_en, nlp_trg=nlp_es)

# Beam 3
(src_tokens, trans_tokens), attns3 = model.translate(src_sentence, max_length=MAX_TRG_LENGTH_TEST, beam_width=3)
utils.show_translation_pair(src_tokens, trans_tokens, nlp_src=nlp_en, nlp_trg=nlp_es)


# Plot attention
head_i = 5
attn = attns1[0][0]  # Remove Beam and Batch
utils.display_attention(src_tokens, trans_tokens[0], attn[head_i], title=f"Attention (Head #{head_i})")

# Plot attention
head_i = 5
attn = attns3[0][0]  # Remove Beam and Batch
utils.display_attention(src_tokens, trans_tokens[0], attn[head_i], title=f"Attention (Head #{head_i})")

print("Done!")
