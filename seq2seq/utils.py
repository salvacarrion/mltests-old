import time
import json
import math

import torch
from torchtext import data
from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm


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


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("test.eps")
    print("save fig!")
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


def calculate_bleu(model, data_iter, max_trg_len, packed_pad=False):
    trgs = []
    trg_pred = []

    model.eval()
    data_iter.batch_size = 1

    for batch in tqdm(data_iter, total=len(data_iter)):
        # Get data
        (src, src_len) = batch.src if packed_pad else (batch.src, None)
        trg = batch.trg

        # Get output
        if packed_pad:
            trg_indexes, _ = model.translate_sentence(src, src_len, max_trg_len)
        else:  # RNN, Transformers
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
