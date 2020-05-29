# Evaluate rationale selection.

prediction_dir=$PWD/prediction/rationale-selection
result_dir=$PWD/results

cd ../..

for name in scifact scifact_only_rationale
do
    rationale_file=$prediction_dir/rationale_selection_dev_rationale_roberta_large_${name}.jsonl
    python verisci/evaluate/rationale_selection.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_dev.jsonl \
        --rationale-selection $rationale_file
done
