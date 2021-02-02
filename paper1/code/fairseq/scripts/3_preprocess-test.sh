#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
SRC_DICT=$4
TRG_DICT=$5
TESTSET_PATH=$6
OUTPUT_PATH=$7
NPROC=$(nproc)

# Fast BPE: https://github.com/glample/fastBPE
FASTBPE_PATH=/home/scarrion/packages/fastBPE/fast

# Show constants
echo "Preprocessing files for Fairseq... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- src dict: "$SRC_DICT
echo "- trg dict: "$TRG_DICT
echo "- Testset path: "$TESTSET_PATH
echo "- Output path: "$OUTPUT_PATH

# Apply BPE
$FASTBPE_PATH applybpe $OUTPUT_PATH/test.tok.bpe.$VOCAB_SIZE.$SRC_LANG $TESTSET_PATH/test.tok.clean.$SRC_LANG $TESTSET_PATH/codes
$FASTBPE_PATH applybpe $OUTPUT_PATH/test.tok.bpe.$VOCAB_SIZE.$TRG_LANG $TESTSET_PATH/test.tok.clean.$TRG_LANG $TESTSET_PATH/codes

# Preprocess files
fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --testpref $OUTPUT_PATH/test.tok.bpe.$VOCAB_SIZE \
    --destdir $OUTPUT_PATH/data-bin/$DATASET \
    --workers	$NPROC \
    --srcdict	$SRC_DICT \
    --tgtdict	$TRG_DICT \




