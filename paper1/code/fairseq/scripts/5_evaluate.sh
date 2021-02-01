#!/bin/sh

# Define constants
SRC_LANG=$1
TRG_LANG=$2
MODEL=$3
DATASET=$4
BASE_PATH=$5
NPROC=$(nproc)
EVAL_PATH="${BASE_PATH}/eval"
BEAM=5
SCORING=bleu

# Show constants
echo "Evaluating model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Model: "$MODEL
echo "- Dataset: "$DATASET
echo "- Beam: "$BEAM
echo "- Scoring method: "$SCORING
echo "- Evaluate path: "$EVAL_PATH
echo "- Base path: "$BASE_PATH


# Create folder
mkdir -p $EVAL_PATH

# Evaluate model
fairseq-generate \
    $BASE_PATH/data-bin/$DATASET/ \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --path $BASE_PATH/checkpoints/$MODEL \
    --tokenizer	moses \
    --remove-bpe \
    --beam $BEAM \
    --scoring $SCORING \
    > "$EVAL_PATH/${DATASET}_${SCORING}_${BEAM}.log"
    #sacrebleu \
#    --quiet	\
