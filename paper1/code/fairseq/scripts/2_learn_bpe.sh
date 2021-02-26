#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4
FASTBPE_PATH=$5

# Fast BPE: https://github.com/glample/fastBPE

# Show constants
echo "Learning BPE... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Create folder
mkdir -p $BASE_PATH/bpe/

# Learn codes (jointly)
$FASTBPE_PATH learnbpe $VOCAB_SIZE $BASE_PATH/clean/train.tok.clean.$SRC_LANG $BASE_PATH/clean/train.tok.clean.$TRG_LANG > $BASE_PATH/bpe/bpecodes
