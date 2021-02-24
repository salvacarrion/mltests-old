#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4
FASTBPE_PATH=$5

# Fast BPE: https://github.com/glample/fastBPE

# Show constants
echo "Applying BPE... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Create folder
mkdir -p $BASE_PATH/bpe/

# Learn codes (jointly)
$FASTBPE_PATH learnbpe $VOCAB_SIZE $BASE_PATH/clean/train.tok.clean.$SRC_LANG $BASE_PATH/clean/train.tok.clean.$TRG_LANG > $BASE_PATH/bpe/bpecodes

# Apply BPE
$FASTBPE_PATH applybpe $BASE_PATH/bpe/train.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/clean/train.tok.clean.$SRC_LANG $BASE_PATH/bpe/bpecodes
$FASTBPE_PATH applybpe $BASE_PATH/bpe/val.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/clean/val.tok.clean.$SRC_LANG $BASE_PATH/bpe/bpecodes
$FASTBPE_PATH applybpe $BASE_PATH/bpe/test.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/clean/test.tok.clean.$SRC_LANG $BASE_PATH/bpe/bpecodes

$FASTBPE_PATH applybpe $BASE_PATH/bpe/train.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/clean/train.tok.clean.$TRG_LANG $BASE_PATH/bpe/bpecodes
$FASTBPE_PATH applybpe $BASE_PATH/bpe/val.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/clean/val.tok.clean.$TRG_LANG $BASE_PATH/bpe/bpecodes
$FASTBPE_PATH applybpe $BASE_PATH/bpe/test.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/clean/test.tok.clean.$TRG_LANG $BASE_PATH/bpe/bpecodes