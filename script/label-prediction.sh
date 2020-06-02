#!/bin/bash
#
# This script recreates the label-prediction results on table 3 of the paper
#
# Usage: bash script/label-prediction.sh [bert-variant] [training-dataset] [dataset]
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
bash script/download-model.sh label ${bert_variant} ${training_dataset}

####################

# Create a prediction folder to store results.
rm -rf prediction
mkdir -p prediction

####################

# Run abstract retrieval.
echo; echo "Retrieving oracle abstracts."
python3 verisci/inference/abstract_retrieval/oracle.py \
    --dataset data/claims_${dataset}.jsonl \
    --include-nei \
    --output prediction/abstract_retrieval.jsonl

####################

# Run rationale selection.
echo; echo "Selecting oracle rationales."
python3 verisci/inference/rationale_selection/oracle_tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --abstract-retrieval prediction/abstract_retrieval.jsonl \
    --output prediction/rationale_selection.jsonl

####################

# Run label prediction
echo; echo "Predicting labels."

if [ training_dataset = "scifact_only_claim" ]
then
    mode="only_claim"
elif [ training_dataset = "scifact_only_rationale" ]
then
    mode="only_rationale"
else
    mode="claim_and_rationale"
fi

python3 verisci/inference/label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --rationale-selection prediction/rationale_selection.jsonl \
    --model model/label_${bert_variant}_${training_dataset} \
    --mode ${mode} \
    --output prediction/label_prediction.jsonl

####################

echo; echo "Evaluating."
python3 verisci/evaluate/label_prediction.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --label-prediction prediction/label_prediction.jsonl