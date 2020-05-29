# Evaluate labeling.

prediction_dir=$PWD/prediction/label-prediction
result_dir=$PWD/results

cd ../..

for name in fever_scifact scifact_only_rationale scifact_only_claim
do
    echo $name
    label_file=$prediction_dir/label_prediction_dev_label_roberta_large_${name}.jsonl
    python verisci/evaluate/label_accuracy_gold_inputs.py \
        --corpus data/corpus.jsonl \
        --dataset data/claims_dev.jsonl \
        --label-prediction $label_file
done
