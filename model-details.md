## Model Download

All the model weights are stored in a public AWS S3 bucket. You can download pre-trained models using the script `script/download-model.sh`. The script usage is as follows:

```bash
bash script/download-model.sh [model-component] [bert-variant] [training-dataset]

# Example usage
bash script/download-model.sh rationale roberta_large scifact
```

There are two `model-compont`s:
1. `rationale`: Identify rationale sentences.
2. `label`: Predict label given input rationales.

There are four `bert-variant`s:
1. `roberta_base`
2. `biomed_roberta_base`
3. `roberta_large`.
5. `scibert`

There are four `training-dataset`s:
1. `fever`
2. `snopes`
3. `scifact`
4. `fever_scifact` (i.e. fever followed by scifact)

The best-performing pipeline reported in Table 1 of the [paper](https://arxiv.org/abs/2004.14974) uses:
- `rationale`: `roberta_large` + `fever`
- `label`: `roberta_large` + `fever_scifact`

For `fever` and `fever_scifact`, there are models available for all 4 BERT variants. For `snopes`, only `roberta_large` is available for download (but you can train your own model).


## Abstract Retrieval

#### Oracle
```sh
python abstract_retrieval/oracle.py \
    --dataset data/claims_dev.jsonl \
    --output abstract_retrieval.jsonl
```
Additional args:
* `--include-nei` (default: False) include documents for `NOT_ENOUGH_INFO` claims.

#### TF-IDF
```sh
python abstract_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output abstract_retrieval.jsonl
```

#### Evaluate
```sh
python abstract_retrieval/evaluate.py \
    --dataset data/claims_test.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl
```

-------------------------------------------------------

## Rationale Selection

#### Oracle
```sh
python rationale_selection/oracle.py \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### Oracle + TF-IDF
Use oracle if gold rationale for that document is present, otherwise use tfidf.
```sh
python rationale_selection/oracle_tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### First Sentence
```sh
python rationale_selection/first.py \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### Last Sentence
```sh
python rationale_selection/last.py \
    --corpus data/corpus.jsonl \
    --abstract-retrieval abstract_retrieval.jsonl \
    --output rationale_selection.jsonl
```

#### TF-IDF
```sh
python rationale_selection/tfidf.py \
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
python rationale_selection/transformer.py \
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
* `--output-k?` and `--output-flex` are optional. Only provide if you need them.

Download models with prefix: `sentence_retrieval` for rationale selection module.

#### Evaluate
```sh
python rationale_selection/evaluate.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl
```

-------------------------------------------------------

## Label Prediction

#### Oracle
```sh
python label_prediction/oracle.py \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl \
    --output label_prediction.jsonl
```

#### Transformers
```sh
python label_prediction/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --rationale-selection rationale_selection.jsonl \
    --model PATH_TO_MODEL \
    --output label_prediction.jsonl
```

Download models with prefix: `nli` for label prediction module.

#### Evaluate
```sh
python label_prediction/evaluate.py \
    --corpus data/corpus.jsonl \
    --dataset data/claim_dev.jsonl \
    --label-prediction label_prediction.jsonl
```

-------------------------------------------------------

## Pipeline

#### Evaluate
```sh
python pipeline/evaluate.py \
    --dataset data/claim_dev.jsonl \
    --rationale-selection rationale_selection.jsonl \
    --label-prediction label_prediction.jsonl
```
