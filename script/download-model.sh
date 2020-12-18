#!/bin/bash
#
# Download pre-trained model from AI2 S3 bucket.
#
# Usage: bash script/download-model.sh [model-component] [bert-variant] [training-dataset]
# [model-component] options: "rationale", "label"
# [bert-variant] options: "roberta_large", "roberta_base", "scibert", "biomed_roberta_base"
# [training-dataset] options: "scifact", "fever_scifact", "fever", "snopes"


model_dir="model"
mkdir -p "$model_dir"

base="https://scifact.s3-us-west-2.amazonaws.com/release/2020-12-17/models"
model_component=$1
bert_variant=$2
training_dataset=$3

model_name="${model_component}_${bert_variant}_${training_dataset}"

# If the file already exists, don't download again.
if [ -e model/$model_name ]
then
    echo "Model $model_name already exists. Skip download."
    exit 0
fi

# Download the compressed tarbell.
wget "$base/${model_name}.tar.gz" -P "$model_dir"
# Extract the tarbell.
tar -xzf "$model_dir/${model_name}.tar.gz" -C "$model_dir"
# Remove the tarball.
rm "$model_dir/${model_name}.tar.gz"
