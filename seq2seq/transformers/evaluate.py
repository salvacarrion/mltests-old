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

###########################################################################
###########################################################################

# Names
EXPERIMENT_NAME = "runs/transformer"
MODEL_NAME = "simple_transformer"

###########################################################################
###########################################################################

# Build model and initialize
DATASET_NAME = "miguel"  # multi30k, miguel
DATASET_PATH = f"../.data/{DATASET_NAME}"
TENSORBOARD = True
ALLOW_DATA_PARALLELISM = False
LEARNING_RATE = 0.0005
MIN_FREQ = 3
MAX_SIZE = 10000 - 4  # 4 reserved words <sos>, <eos>, <pad>, <unk>
N_EPOCHS = 1000
MAX_SRC_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
MAX_TRG_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
MAX_TRG_LENGTH_TEST = int(MAX_TRG_LENGTH * 1.0)  # len>1.0 is not supported by all models
BATCH_SIZE = 32
CHECKPOINT_PATH = f'checkpoints/checkpoint_{MODEL_NAME}.pt'
TR_RATIO = 1.0
DV_RATIO = 1.0
TS_RATIO = 1.0
TB_BATCH_RATE = 100
SOS_WORD = '<sos>'
EOS_WORD = '<eos>'
EVALUATE = True
BLUE = True

###########################################################################
###########################################################################

print("###########################################################################")
print("###########################################################################")
print(f"- Mode: Evaluation")
print(f"- Executing model: {EXPERIMENT_NAME}")
print(f"- Experiment name: {MODEL_NAME}")
print(f"- Checkpoint path: {CHECKPOINT_PATH}")
print("###########################################################################")
print("###########################################################################")

###########################################################################
###########################################################################

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###########################################################################
###########################################################################

# Set fields
SRC = data.Field(tokenize='spacy', tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
TRG = data.Field(tokenize='spacy', tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, batch_first=True)
fields = [('src', SRC), ('trg', TRG)]

# Load examples
train_data = utils.load_dataset(f"{DATASET_PATH}/tokenized/train.json", fields, TR_RATIO)
test_data = utils.load_dataset(f"{DATASET_PATH}/tokenized/test.json", fields, TS_RATIO)

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

start = time.time()

# Build vocab
SRC.build_vocab(train_data, max_size=MAX_SIZE)
TRG.build_vocab(train_data, max_size=MAX_SIZE)

end = time.time()
print(f"Time to build vocabularies: {end - start} sec")

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (es) vocabulary: {len(TRG.vocab)}")

###########################################################################
###########################################################################

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(utils.gpu_info())

# Set iterator (this is where words are replaced by indices, and <sos>/<eos> tokens are appended
train_iter, test_iter = data.BucketIterator.splits((train_data, test_data),
                                                   batch_size=BATCH_SIZE, device=device, sort=False)

###########################################################################
###########################################################################

# Select model
if MODEL_NAME == "simple_transformer":
    from seq2seq.models import s2s_6_transfomer as builder
    model = builder.make_model(src_field=SRC, trg_field=TRG,
                               max_src_len=MAX_SRC_LENGTH, max_trg_len=MAX_TRG_LENGTH, device=device,
                               data_parallelism=ALLOW_DATA_PARALLELISM)
    model.apply(builder.init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

else:
    raise ValueError("Unknown model name")

###########################################################################
###########################################################################

# Set loss (ignore when the target token is <pad>)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

###########################################################################
###########################################################################

# Load best model
model.load_state_dict(torch.load(CHECKPOINT_PATH))
print("Model loaded!")

# Evaluate best model
if EVALUATE:
    start = time.time()
    test_loss = helpers.evaluate(model, test_iter, criterion)
    helpers.summary_report(test_loss=test_loss, start_time=start)

# Calculate BLEU score
if BLUE:
    start = time.time()
    bleu_score = utils.calculate_bleu(model, test_iter, max_trg_len=MAX_TRG_LENGTH_TEST)

    end_time = time.time()
    epoch_mins, epoch_secs = utils.epoch_time(start, end_time)
    print(f'BLEU score = {bleu_score * 100:.2f} | Time: {epoch_mins}m {epoch_secs}s')

###########################################################################
###########################################################################

print("Done!")
