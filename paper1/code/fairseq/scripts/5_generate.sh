#!/bin/sh

# Define constants
SRC_LANG=$1
TRG_LANG=$2
TESTSET=$3
MODEL_PATH=$4
BASE_PATH=$5
DATASET=$6
NPROC=$(nproc)
EVAL_PATH="${BASE_PATH}/eval/${TESTSET}"
BEAM=5
SCORING=bleu

# Show constants
echo "Evaluating model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Testset: "$TESTSET
echo "- Dataset: "$DATASET
echo "- Beam: "$BEAM
echo "- Scoring method: "$SCORING
echo "- Model path: "$MODEL_PATH
echo "- Evaluate path: "$EVAL_PATH
echo "- Base path: "$BASE_PATH


# Create folder
mkdir -p $EVAL_PATH

# Evaluate model
fairseq-generate \
    $BASE_PATH/data-bin \
    --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --path $MODEL_PATH \
    --tokenizer	moses \
    --remove-bpe \
    --beam $BEAM \
    --scoring $SCORING \
    --results-path $EVAL_PATH \
#    > "$EVAL_PATH/${DATASET}_${SCORING}_${BEAM}.log"
    #sacrebleu \
#    --quiet	\


#fairseq-generate data-bin/scielo_health_es_en/ --source-lang es --target-lang en --path checkpoints/transformer/checkpoint_best.pt --tokenizer moses --remove-bpe --beam 5 --scoring bleu
#fairseq-interactive data-bin/scielo_health_es_en/ --path checkpoints/transformer/checkpoint_best.pt --beam 5 --source-lang es --target-lang en --tokenizer moses --bpe fastbpe --bpe-codes codes