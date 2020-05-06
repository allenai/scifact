# Download pre-trained model from AI2 S3 bucket.
# Usage: bash script/download-model.sh [model-component] [bert-variant] [training-dataset]
# Example: bash script/download-model.sh rationale roberta_base fever_scifact

# See `baselines.md` for more details on the available models.

model_dir="model"
mkdir -P "$model_dir"

base="https://ai2-s2-scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/models"
model_component=$1
bert_variant=$2
training_dataset=$3

wget "$base/${model_component}_${bert_variant}_${training_dataset}.tar.gz" -P "$model_dir"
