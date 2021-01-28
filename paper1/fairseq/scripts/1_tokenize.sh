#!/bin/sh

# Define constants
SRC_LANG=$1
TRG_LANG=$2
BASE_PATH=$3
NPROC=$(nproc)

# Sacremoses: https://github.com/alvations/sacremoses

# Show constants
echo "Tokenizing files... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Learn codes (jointly)..................
echo "Tokenizing source files..."
sacremoses -l $SRC_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/train.$SRC_LANG > $BASE_PATH/train.tok.clean.$SRC_LANG
sacremoses -l $SRC_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/val.$SRC_LANG > $BASE_PATH/val.tok.clean.$SRC_LANG
sacremoses -l $SRC_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/test.$SRC_LANG > $BASE_PATH/test.tok.clean.$SRC_LANG

echo "Tokenizing target files..."
sacremoses -l $TRG_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/train.$TRG_LANG > $BASE_PATH/train.tok.clean.$TRG_LANG
sacremoses -l $TRG_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/val.$TRG_LANG > $BASE_PATH/val.tok.clean.$TRG_LANG
sacremoses -l $TRG_LANG -j $NPROC normalize -c tokenize -a < $BASE_PATH/test.$TRG_LANG > $BASE_PATH/test.tok.clean.$TRG_LANG

