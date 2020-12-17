#!/bin/bash
#
# This script recreates table 4 of the paper.
#
# Usgae: bash script/pipeline.sh [retrieval] [model] [dataset]
# [retrieval] options: "oracle", "open"
# [model] options: "oracle-rationle", "zero-shot", "verisci"
# [dataset] options: "dev", "test"

retrieval=$1
model=$2
dataset=$3

if [ $model = "zero-shot" ]
then
    rationale_training_dataset="fever"
    rationale_threshold=0.025
    label_training_dataset="fever"
else
    rationale_training_dataset="scifact"
    rationale_threshold=0.5
    label_training_dataset="fever_scifact"
fi

echo "Running pipeline on ${dataset} set."

####################

# Download data.
bash script/download-data.sh

####################

# Download models.
bash script/download-model.sh rationale roberta_large ${rationale_training_dataset}
bash script/download-model.sh label roberta_large ${label_training_dataset}

####################

# Create a prediction folder to store results.
rm -rf prediction
mkdir -p prediction

####################

# Run abstract retrieval.
echo; echo "Retrieving abstracts."
if [ $retrieval == "oracle" ]
then
    python3 verisci/inference/abstract_retrieval/oracle.py \
        --dataset data/claims_${dataset}.jsonl \
        --output prediction/abstract_retrieval.jsonl
else
    python3 verisci/inference/abstract_retrieval/tfidf.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_${dataset}.jsonl \
        --k 3 \
        --min-gram 1 \
        --max-gram 2 \
        --output prediction/abstract_retrieval.jsonl
fi

####################

# Run rationale selection
echo; echo "Selecting rationales."
if [ $model == "oracle-rationale" ]
then
    python3 verisci/inference/rationale_selection/oracle.py \
        --dataset data/claims_${dataset}.jsonl \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --output prediction/rationale_selection.jsonl
else
    python3 verisci/inference/rationale_selection/transformer.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_${dataset}.jsonl \
        --threshold ${rationale_threshold} \
        --abstract-retrieval prediction/abstract_retrieval.jsonl \
        --model model/rationale_roberta_large_${rationale_training_dataset}/ \
        --output-flex prediction/rationale_selection.jsonl
fi

####################

# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
python3 verisci/inference/label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_${dataset}.jsonl \
    --rationale-selection prediction/rationale_selection.jsonl \
    --model model/label_roberta_large_${label_training_dataset} \
    --output prediction/label_prediction.jsonl

####################

# Merge rationale and label predictions.
echo; echo "Merging predictions."
python3 verisci/inference/merge_predictions.py \
    --rationale-file prediction/rationale_selection.jsonl \
    --label-file prediction/label_prediction.jsonl \
    --result-file prediction/merged_predictions.jsonl


####################


# Evaluate final predictions
echo; echo "Evaluating."
python3 verisci/evaluate/pipeline.py \
    --gold data/claims_${dataset}.jsonl \
    --corpus data/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl
