#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
DATASET=$4
BASE_PATH=$5

# Show constants
echo "Preprocessing files for Fairseq... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Dataset: "$DATASET
echo "- Base path: "$BASE_PATH

# Preprocess files
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --trainpref $BASE_PATH/bpe/train.tok.bpe.$VOCAB_SIZE \
    --validpref $BASE_PATH/bpe/val.tok.bpe.$VOCAB_SIZE \
    --testpref $BASE_PATH/bpe/test.tok.bpe.$VOCAB_SIZE \
    --destdir $BASE_PATH/data-bin \
    --workers	$(nproc) \
#    --srcdict	$BASE_PATH/dict/dict.${SRC_LANG}.txt \
#    --tgtdict	$BASE_PATH/dict/dict.${TRG_LANG}.txt \

