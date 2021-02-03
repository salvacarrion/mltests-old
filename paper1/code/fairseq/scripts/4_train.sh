#!/bin/sh

# Define constants
SRC_LANG=$1
TRG_LANG=$2
BASE_PATH=$3

# Show constants
echo "Training model... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Train model
fairseq-train \
     $BASE_PATH/data-bin/ \
     --arch transformer \
     --dropout 0.1 --weight-decay 0.0001 \
     --optimizer adam \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --max-epoch	50 \
     --update-freq 8 \
     --max-tokens 4096 \
     --warmup-updates 4000 \
     --seed 1 \
     --lr 1e-3  \
     --lr-scheduler reduce_lr_on_plateau  \
     --clip-norm 1.0 \
     --no-epoch-checkpoints \
     --log-format simple \
     --save-dir $BASE_PATH/checkpoints \
     --best-checkpoint-metric bleu \
     --tensorboard-logdir $BASE_PATH/logdir \
     --task translation \
     --eval-bleu \
     --eval-bleu-args '{"beam": 5}' \
     --eval-bleu-detok moses \
     --eval-bleu-remove-bpe \
     --eval-bleu-print-samples \
     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
     --patience	5
     --num-workers $(nproc) \


#    --force-anneal 50 \
#    --fp16 \
#    --finetune-from-model "/home/salvacarrion/Documents/Programming/Datasets/Scielo/backups/local_01_29/checkpoints_scielo_health_es_en/transformer/checkpoint_best.pt" \


#fairseq-preprocess --source-lang es --target-lang en --trainpref train.tok.bpe.32000 --validpref val.tok.bpe.32000 --testpref test.tok.bpe.32000 --workers $(nproc)
#fairseq-train data-bin/ --arch transformer --optimizer adam --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --num-workers $(nproc) --max-epoch 50 --seed 1 --log-format simple --keep-last-epochs 2 --keep-best-checkpoints 2 --update-freq 8 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --clip-norm 1.0 --lr 1e-3  --lr-scheduler reduce_lr_on_plateau  --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001
#fairseq-generate data-bin/ --source-lang es --target-lang en --path checkpoints/checkpoint_best.pt --tokenizer moses --remove-bpe --beam 5 --scoring bleu