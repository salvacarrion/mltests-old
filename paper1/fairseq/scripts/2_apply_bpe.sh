#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
BASE_PATH=$4

# Fast BPE: https://github.com/glample/fastBPE
FASTBPE_PATH=/home/salvacarrion/Documents/packages/fastBPE/fast

# Show constants
echo "Applying BPE... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Learn codes (jointly)
$FASTBPE_PATH learnbpe $VOCAB_SIZE $BASE_PATH/train.tok.clean.$SRC_LANG $BASE_PATH/train.tok.clean.$TRG_LANG > $BASE_PATH/codes

# Apply BPE
$FASTBPE_PATH applybpe $BASE_PATH/train.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/train.tok.clean.$SRC_LANG $BASE_PATH/codes
$FASTBPE_PATH applybpe $BASE_PATH/val.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/val.tok.clean.$SRC_LANG $BASE_PATH/codes
$FASTBPE_PATH applybpe $BASE_PATH/test.tok.bpe.$VOCAB_SIZE.$SRC_LANG $BASE_PATH/test.tok.clean.$SRC_LANG $BASE_PATH/codes

$FASTBPE_PATH applybpe $BASE_PATH/train.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/train.tok.clean.$TRG_LANG $BASE_PATH/codes
$FASTBPE_PATH applybpe $BASE_PATH/val.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/val.tok.clean.$TRG_LANG $BASE_PATH/codes
$FASTBPE_PATH applybpe $BASE_PATH/test.tok.bpe.$VOCAB_SIZE.$TRG_LANG $BASE_PATH/test.tok.clean.$TRG_LANG $BASE_PATH/codes

# Get train vocabulary
$FASTBPE_PATH getvocab $BASE_PATH/train.tok.bpe.$VOCAB_SIZE.$SRC_LANG > $BASE_PATH/vocab.tok.bpe.$VOCAB_SIZE.$SRC_LANG
$FASTBPE_PATH getvocab $BASE_PATH/train.tok.bpe.$VOCAB_SIZE.$TRG_LANG > $BASE_PATH/vocab.tok.bpe.$VOCAB_SIZE.$TRG_LANG
