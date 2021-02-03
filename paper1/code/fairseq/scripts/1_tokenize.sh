#!/bin/sh

# Define constants
SRC_LANG=$1
TRG_LANG=$2
BASE_PATH=$3

# Sacremoses: https://github.com/alvations/sacremoses

# Show constants
echo "Tokenizing files... ****************"
echo "- Source language: "$SRC_LANG
echo "- Target language: "$TRG_LANG
echo "- Base path: "$BASE_PATH

# Create folder
mkdir -p $BASE_PATH/clean/

# Learn codes (jointly)..................
echo "Tokenizing source files..."
sacremoses -l $SRC_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/train.$SRC_LANG > $BASE_PATH/clean/train.tok.clean.$SRC_LANG
sacremoses -l $SRC_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/val.$SRC_LANG > $BASE_PATH/clean/val.tok.clean.$SRC_LANG
sacremoses -l $SRC_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/test.$SRC_LANG > $BASE_PATH/clean/test.tok.clean.$SRC_LANG

echo "Tokenizing target files..."
sacremoses -l $TRG_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/train.$TRG_LANG > $BASE_PATH/clean/train.tok.clean.$TRG_LANG
sacremoses -l $TRG_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/val.$TRG_LANG > $BASE_PATH/clean/val.tok.clean.$TRG_LANG
sacremoses -l $TRG_LANG -j $(nproc) normalize -c tokenize -a < $BASE_PATH/raw/test.$TRG_LANG > $BASE_PATH/clean/test.tok.clean.$TRG_LANG

