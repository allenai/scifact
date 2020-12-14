# Models

As described in the paper, the model consists of three pipeline components:
abstract retrieval, rationale selection, and label prediction. We also run
"oracle" experiments where pipeline components are replaced with oracles that
always make correct predictions given correct outputs. The examples below
show how to make oracle and model predictions, evaluate each pipeline component.

## Outline

- [Abstract Retrieval](#abstract-retrieval)
- [Rationale Selection](#rationale-selection)
- [Label Prediction](#label-prediction)
- [Full pipeline](#full-pipeline)


## Abstract Retrieval

#### Oracle
```sh
python verisci/inference/abstract_retrieval/oracle.py \
    --dataset data/claims_dev.jsonl \
    --output abstract_retrieval.jsonl
```
Additional args:
* `--include-nei` (default: False) include documents for `NOT_ENOUGH_INFO` claims.

#### TF-IDF
```sh
python verisci/inference/abstract_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output abstract_retrieval.jsonl
```

#### Evaluate
```sh
python verisci/evalaute/abstract_retrieval.py \
    --dataset data/claims_test.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl
```

-------------------------------------------------------

## Rationale Selection

#### Oracle
```sh
python verisci/inference/rationale_selection/oracle.py \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### Oracle + TF-IDF
Use oracle if gold rationale for that document is present, otherwise use tfidf.
```sh
python verisci/inference/rationale_selection/oracle_tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### First Sentence
```sh
python verisci/inference/rationale_selection/first.py \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### Last Sentence
```sh
python verisci/inference/rationale_selection/last.py \
    --corpus data/corpus.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### TF-IDF
```sh
python verisci/inference/rationale_selection/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --min-gram 1 \
    --max-gram 1 \
    --k 2 \
    --output rationale_selection.jsonl
```

#### Transformer
```sh
python verisci/inference/rationale_selection/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --model PATH_TO_MODEL \
    --output-flex rationale_selection_flex.jsonl \
    --output-k2 rationale_selection_k2.jsonl \
    --output-k3 rationale_selection_k3.jsonl \
    --output-k4 rationale_selection_k4.jsonl \
    --output-k5 rationale_selection_k5.jsonl
```
Additional args:
* `--threshold` (default: 0.5) the threshold for flex output
* `--only-rationale` only pass abstract sentences to the model without claim
* `--output-*` are optional. Only provide them if you need them.

#### Evaluate
```sh
python verisci/evaluate/rationale_selection.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl
```

-------------------------------------------------------

## Label Prediction

#### Oracle
```sh
python verisci/inference/label_prediction/oracle.py \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl \
    --output label_prediction.jsonl
```

#### Transformers
```sh
python verisci/inference/label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl \
    --model PATH_TO_MODEL \
    --output label_prediction.jsonl
```
* `--mode` dictates what to pass in the model, options: `claims_and_rationale` (default), `only_claim`, `only_rationale`

#### Evaluate
```sh
python verisci/evaluate/label_prediction.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --label-prediction label_prediction.jsonl
```

-------------------------------------------------------

## Full Pipeline

#### Merge label and rationale predictions
```sh
python verisci/inference/merge_predictions.py \
    --rationale-file prediction/rationale_selection.jsonl \
    --label-file prediction/label_prediction.jsonl \
    --result-file prediction/merged_predictions.jsonl
```

#### Evaluate the merged predictions
```sh
python verisci/evaluate/pipeline.py \
    --gold data/claims_dev.jsonl \
    --corpus data/corpus.jsonl \
    --prediction prediction/merged_predictions.jsonl \
    --output predictions/metrics.json   # If no `output`, provided, print to console.
```
