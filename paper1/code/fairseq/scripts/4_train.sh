#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
DATASET=$4
BASE_PATH=$5
NPROC=$(nproc)

# Show constants
echo "Training model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Dataset: "$DATASET
echo "- Base path: "$BASE_PATH

# Train model
fairseq-train \
    $BASE_PATH/data-bin/$DATASET \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --lr-scheduler reduce_lr_on_plateau \
    --num-workers	$NPROC \
    --max-epoch	15 \
    --seed 1 \
    --save-dir $BASE_PATH/checkpoints/transformer \
    --keep-last-epochs 2 \
    --keep-best-checkpoints 2 \
    --tensorboard-logdir $BASE_PATH/logdir \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#    --finetune-from-model "/home/salvacarrion/Documents/Programming/Datasets/Scielo/backups/local_01_29/checkpoints_scielo_health_es_en/transformer/checkpoint_best.pt" \


