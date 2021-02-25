import torch
import fairseq
from fairseq import checkpoint_utils
from fairseq.models.transformer import TransformerModel

# Constants
BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq"
DATASET = "scielo_health_es_en"
CHECKPOINTS_PATH = f"{BASE_PATH}/{DATASET}"
DATA_PATH = f"{BASE_PATH}/{DATASET}/data-bin/{DATASET}"
BPE_CODES = f"{BASE_PATH}/{DATASET}/codes"

# Load model
model = TransformerModel.from_pretrained(
  "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/scielo_health_es_en/checkpoints",
  checkpoint_file='transformer/checkpoint_best.pt',
  data_name_or_path="/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/scielo_health_es_en/data-bin/scielo_health_es_en",
  tokenizer='moses',
  bpe='fastbpe',
  bpe_codes="/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/scielo_health_es_en/codes",
  eval_bleu_detok_args='{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'
)
tmp = model.translate("Hola cómo estas?")
sdas = 3