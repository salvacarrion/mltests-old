#!/bin/sh

# Define constants
VOCAB_SIZE=$1
SRC_LANG=$2
TRG_LANG=$3
MODEL=$4
DATASET=$5
BASE_PATH=$6
NPROC=$(nproc)
EVAL_PATH=$BASE_PATH/eval
BEAM=5
SCORING=bleu

# Create folder
mkdir -p EVAL_PATH

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

# Evaluate model
fairseq-generate \
    $BASE_PATH/data-bin/$DATASET/ \
    --path $BASE_PATH/checkpoints/$MODEL \
    --tokenizer	moses \
    --remove-bpe \
    --beam $BEAM \
    --scoring	$SCORING \
    > "${BASE_PATH}/eval/${DATASET}_${SCORING}_${BEAM}"
    #sacrebleu \
#    --quiet	\
