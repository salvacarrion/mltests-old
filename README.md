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
DATASET_NANE=...
TEXT=examples/translation/$DATASET_NANE
fairseq-preprocess --source-lang $src --target-lang $trg --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test --destdir data-bin/$DATASET_NANE
```


3. Preprocessed data (fairseq)

```
```