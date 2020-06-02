#!/bin/bash
#
# This script recreates the rationale-selection results on table 3 of the paper
#
# Usage: bash script/rationale-selection.sh [bert-variant] [training-dataset] [dataset]
# [bert-variant] options: "roberta_large", "roberta_base", "scibert", "biomed_roberta_base"
# [training-dataset] options: "scifact", "fever_scifact", "fever", "snopes"
# [dataset] options: "dev", "test"

bert_variant=$1
training_dataset=$2
dataset=$3

echo "Running pipeline on ${dataset} set."

####################

# Download data.
bash script/download-data.sh

####################

# Download models.
bash script/download-model.sh rationale ${bert_variant} ${training_dataset}

####################

# Create a prediction folder to store results.
rm -rf prediction
mkdir -p prediction

####################

# Run abstract retrieval.
echo; echo "Retrieving oracle abstracts."
python3 verisci/inference/abstract_retrieval/oracle.py \
    --dataset data/claims_${dataset}.jsonl \
    --output prediction/abstract_retrieval.jsonl

####################

# Run rationale selection, using the retrieved oracle abstracts. Use `flex` selection,
# which selects any rationale with a score >= 0.5
echo; echo "Selecting rationales."
python3 verisci/inference/rationale_selection/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --abstract-retrieval prediction/abstract_retrieval.jsonl \
    --model model/rationale_${bert_variant}_${training_dataset}/ \
    --output-flex prediction/rationale_selection.jsonl

####################

echo; echo "Evaluating."
python3 verisci/evaluate/rationale_selection.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --rationale-selection prediction/rationale_selection.jsonl
