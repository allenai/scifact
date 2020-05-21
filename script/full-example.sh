# Full example showing how to download the models, make predictions, and
# evaluate. This script should reproduce the final numbers in Table 1 of the
# paper.

# Usgae: bash script/full-example.sh

####################

# Download models.
bash script/download-model.sh rationale roberta_large scifact
bash script/download-model.sh label roberta_large fever_scifact

####################

# Create a prediction folder to store results.
mkdir prediction

####################

# Run abstract retrieval.
echo; echo "Retrieving abstracts."
python abstract_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output prediction/abstract_retrieval_dev.jsonl

####################

# Run rationale selection, using the retrieved abstracts. Use `flex` selection,
# which selects any rationale with a score of >+ 0.5
echo; echo "Selecting rationales."
python rationale_selection/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval prediction/abstract_retrieval_dev.jsonl \
    --model model/rationale_roberta_large_scifact/ \
    --output-flex prediction/rationale_selection_dev.jsonl \

####################

# Run label prediction, using the selected rationales.
echo; echo "Predicting labels."
python label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --rationale-selection prediction/rationale_selection_dev.jsonl \
    --model model/label_roberta_large_fever_scifact \
    --output prediction/label_prediction_dev.jsonl

####################

# Evaluate final predictions
echo; echo "Evaluating."
python pipeline/evaluate_paper_metrics.py \
    --dataset data/claims_dev.jsonl \
    --corpus data/corpus.jsonl \
    --rationale-selection prediction/rationale_selection_dev.jsonl \
    --label-prediction prediction/label_prediction_dev.jsonl
