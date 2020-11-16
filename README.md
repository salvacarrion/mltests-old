# MLTests

Backup for my ML tests

### Useful commands

```
python -m pip install --upgrade pip
pip install -U --use-feature=2020-resolver -r requirements.txt 
tensorboard --logdir ./lightning_logs
```

## Fairseq

1. Run "fairseq/preprocessed"

2. Apply FastBPE:

```
size=32000
src=en
trg=es
fast learnbpe $size train.tok.$src train.tok.$trg > codes

fast applybpe train.$size.$src train.$src codes
fast applybpe train.$size.$trg train.$trg codes

fast applybpe dev.$size.$src dev.$src codes
fast applybpe dev.$size.$trg dev.$trg codes

fast applybpe test.$size.$src test.$src codes
fast applybpe test.$size.$trg test.$trg codes
```

3. Preprocessed data (fairseq)

```
DATASET_NANE=miguel
TEXT=examples/translation/$DATASET_NANE
fairseq-preprocess --source-lang $src --target-lang $trg --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test --destdir data-bin/$DATASET_NANE --tokenizer moses --bpe fastbpe --workers 16
```


4. Train (fairseq)

```
CHECKPOINTS_PATH=checkpoints/fconv/
mkdir -p $CHECKPOINTS_PATH
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/$DATASET_NANE --lr 0.001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --optimizer adam --lr-scheduler reduce_lr_on_plateau --tensorboard-logdir runs/ --num-workers 8 --log-format tqdm --save-dir $CHECKPOINTS_PATH --no-epoch-checkpoints
```

5. Interact (fairseq)

```
# Create new folder a copy to it, these files: checkpoint_best (checkpoints), codes (examples), dict.src.txt, dict.trg.txt (data-bin)
$MODEL_DIR=...
fairseq-interactive --path $MODEL_DIR/checkpoint_best.pt $MODEL_DIR --beam 5 --source-lang $src --target-lang $trg --tokenizer moses --bpe fastbpe --bpe-codes $MODEL_DIR/codes
```

6. Generate (fairseq)

```
fairseq-generate data-bin/$DATASET_NANE --path $CHECKPOINTS_PATH/checkpoint_best.pt --batch-size 128 --beam 5
```

