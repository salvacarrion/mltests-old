import time
import json
import math
import pickle

import torch
from torchtext import data
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import seaborn as sns
# sns.set() Problems with attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_info():
    if torch.cuda.is_available():
        s = f"- Using GPU: {torch.cuda.is_available()}\n" \
            f"- No. devices: {torch.cuda.device_count()}\n" \
            f"- Device name (0): {torch.cuda.get_device_name(0)}"
    else:
        s = "- Using CPU"
    return s


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def display_attention(sentence, translation, attention, savepath=None, title="Attention", ax=None):
    if ax is not None:
        _ax = ax
    else:
        fig, _ax = plt.subplots()

    cax = _ax.matshow(attention, cmap='bone')

    _ax.tick_params(labelsize=15)
    _ax.set_xticklabels([''] + sentence, rotation=45)
    _ax.set_yticklabels([''] + translation)

    _ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    _ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    _ax.set_title(title)
    _ax.set_xlabel("Source")
    _ax.set_ylabel("Translation")

    plt.rcParams['figure.figsize'] = [10, 10]

    # Save figure
    if savepath:
        plt.savefig(savepath)
        print("save attention!")

    if ax is not None:
        plt.show()
        plt.close()


def save_dataset_examples(dataset, savepath):
    start = time.time()

    total = len(dataset.examples)
    with open(savepath, 'w') as f:
        # Save num. elements
        f.write(json.dumps(total))
        f.write("\n")

        # Save elements
        for pair in tqdm(dataset.examples, total=total):
            data = [pair.src, pair.trg]
            f.write(json.dumps(data))
            f.write("\n")

    end = time.time()
    print(f"Save dataset examples: [Total time= {end - start}; Num. examples={total}]")


def load_dataset(filename, fields, ratio=1.0):
    start = time.time()

    examples = []
    with open(filename, 'rb') as f:
        # Read num. elements
        line = f.readline()
        total = json.loads(line)

        # Load elements
        limit = int(total * ratio)
        for i in tqdm(range(limit), total=limit):
            line = f.readline()
            example = json.loads(line)
            example = data.Example().fromlist(example, fields)  # Create Example obj.
            examples.append(example)

    # Build dataset ("examples" passed by reference)
    dataset = data.Dataset(examples, fields)

    end = time.time()
    print(f"Load dataset: [Total time= {end - start}; Num. examples={len(dataset.examples)}]")
    return dataset


def load_vocabulary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_vocabulary(field, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(field.vocab, f)


def calculate_bleu(model, data_iter, max_trg_len, packed_pad=False):
    trgs = []
    trg_pred = []

    model.eval()
    data_iter.batch_size = 1

    for batch in tqdm(data_iter, total=len(data_iter)):
        # Get output
        if packed_pad:
            (src, src_len), (trg, trg_len) = batch.src, batch.trg
            trg_indexes, _ = model.translate_sentence(src, src_len, max_trg_len)
        else:  # RNN, Transformers
            src, trg = batch.src, batch.trg
            trg_indexes, _ = model.translate_sentence(src, max_trg_len)

        # Convert predicted indices to tokens
        trg_pred_tokens = [model.trg_field.vocab.itos[i] for i in trg_indexes]
        trg_tokens = [model.trg_field.vocab.itos[i] for i in trg.detach().cpu().int().flatten()]

        # Remove special tokens
        trg_pred_tokens = trg_pred_tokens[1:-1]  # Remove <sos> and <eos>
        trg_tokens = trg_tokens[1:-1]  # Remove <sos> and <eos>

        # Add predicted token
        trg_pred.append(trg_pred_tokens)
        trgs.append([trg_tokens])

    # Compute score
    score = bleu_score(trg_pred, trgs)
    return score


