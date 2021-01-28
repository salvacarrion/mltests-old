#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4
NPROC=$(nproc)

# Show constants
echo "Training model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Train model
fairseq-train \
    $BASE_PATH/data-bin \
    --arch transformer \
    --criterion cross_entropy \
    --optimizer	adam \
    --scoring bleu \
    --lr 0.25 \
    --lr-scheduler reduce_lr_on_plateau \
    --num-workers	$NPROC \
    --max-tokens 4000 \
    --max-epoch	10 \
    --seed 1 \
    --save-dir $BASE_PATH/checkpoints/transformer \
    --tensorboard-logdir $BASE_PATH/logdir \

