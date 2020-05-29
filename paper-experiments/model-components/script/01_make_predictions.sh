# Usage: bash script/01_make_predictions.sh
# Invoke from the `table-3` directory.


####################

prediction_dir=$PWD/prediction

predict_rationales() {
    rationale_selector=$1
    python verisci/inference/rationale_selection/transformer.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_dev.jsonl \
        --abstract-retrieval $prediction_dir/rationale-selection/abstract_retrieval_dev_oracle.jsonl \
        --model model/$rationale_selector \
        --output-flex $prediction_dir/rationale-selection/rationale_selection_dev_${rationale_selector}.jsonl
}

cd ../..

# Do rationale selection
python verisci/inference/abstract_retrieval/oracle.py \
    --dataset data/claims_dev.jsonl \
    --output $prediction_dir/rationale-selection/abstract_retrieval_dev_oracle.jsonl

for rationale_selector in rationale_roberta_large_scifact rationale_roberta_large_scifact_only_rationale
do
    predict_rationales $rationale_selector
done

####################


# predict_labels() {
#     retrieval_file=$1
#     rationale_file=$2
#     label_predictor=$3
#     output_file=$prediction_dir/label-prediction/label_prediction_dev_${label_predictor}.jsonl

#     python verisci/inference/label_prediction/transformer.py \
#         --corpus data/corpus.jsonl \
#         --dataset data/claims_dev.jsonl \
#         --model model/$label_predictor \
#         --rationale-selection $prediction_dir/label-prediction/rationale_selection_dev_oracle.jsonl \
#         --output $output_file
# }


# # Do label prediction

# prediction_dir=$PWD/prediction
# cd ../..

# retrieval_file=$prediction_dir/label-prediction/abstract_retrieval_dev_oracle.jsonl
# rationale_file=$prediction_dir/label-prediction/rationale_selection_dev_oracle.jsonl

# python verisci/inference/abstract_retrieval/oracle.py \
#     --dataset data/claims_dev.jsonl \
#     --output $retrieval_file \
#     --include-nei

# python verisci/inference/rationale_selection/oracle_tfidf.py \
#     --corpus data/corpus.jsonl \
#     --dataset data/claims_dev.jsonl \
#     --abstract-retrieval $retrieval_file \
#     --output $rationale_file

# for label_predictor in label_roberta_large_fever_scifact label_roberta_large_scifact_only_claim label_roberta_large_scifact_only_rationale
# do
#     predict_labels $retrieval_file $rationale_file $label_predictor
# done
