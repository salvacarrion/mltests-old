import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

import math
import re
import unicodedata

# Define regex patterns
p_whitespace = re.compile(" +")


def preprocess_text(text):
    try:
        # Remove repeated whitespaces "   " => " "
        text = p_whitespace.sub(' ', text)

        # Normalization Form Compatibility Composition
        text = unicodedata.normalize("NFD", text)

        # Strip whitespace
        text = text.strip()
    except TypeError as e:
        # print(f"=> Error preprocessing: '{text}'")
        text = ""
    return text


def preprocess_dataset(df, src_col, trg_col):
    to_remove = [False]*len(df)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        src, trg = row[src_col], row[trg_col]

        # Preprocess
        src = preprocess_text(src)
        trg = preprocess_text(trg)

        # Remove empty line
        if len(src) == 0 or len(trg) == 0:
            to_remove[i] = True
        # elif math.fabs(len(src)-len(trg)) > 20:
        #     to_remove[i] = True

    # Remove rows
    df = df.drop(df.index[to_remove])
    return df


def human_format(num, idx=None):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
