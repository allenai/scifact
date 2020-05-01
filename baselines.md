## Model Download

All the model weights are stored separately using Google drive. You can download the corresponding
model tar.gz and uncompress it in the `/models` directory

[Download Models](https://drive.google.com/drive/folders/1ni0dI7dhYDsKFoz6mfdEZD3ibSawq3mY?usp=sharing)

## Document Retrieval

#### Oracle
```sh
python document_retrieval/oracle.py \
    --dataset data/dataset.test.jsonl \
    --output document_retrieval.jsonl
```

#### TF-IDF
```sh
python document_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --k 3 \
    --min-gram 1 \
    --max-gram 2 \
    --output document_retrieval.jsonl
```

#### Evaluate
```sh
python document_retrieval/evaluate.py \
    --dataset data/dataset.test.jsonl \
    --document-retrieval document_retrieval.jsonl
```

-------------------------------------------------------

## Sentence Retrieval

#### Oracle
```sh
python sentence_retrieval/oracle.py \
    --dataset data/dataset.test.jsonl \
    --document-retrieval document_retrieval.jsonl \
    --output sentence_retrieval.jsonl
```

#### Oracle + TF-IDF
```sh
python sentence_retrieval/oracle_tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --output sentence_retrieval.jsonl
```

#### First Sentence
```sh
python sentence_retrieval/first.py \
    --document-retrieval document_retrieval.jsonl \
    --output sentence_retrieval.jsonl
```

#### Last Sentence
```sh
python sentence_retrieval/last.py \
    --corpus data/corpus.jsonl \
    --document-retrieval document_retrieval.jsonl \
    --output sentence_retrieval.jsonl
```

#### TF-IDF
```sh
python sentence_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --document-retrieval document_retrieval.jsonl \
    --min-gram 1 \
    --max-gram 1 \
    --k 2 \
    --output sentence_retrieval.jsonl
```

#### Transformer
```sh
python sentence_retrieval/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --document-retrieval document_retrieval.jsonl \
    --model models/sentence_retrieval_scibert_fever_pretraining \
    --flexk sentence_retrieval_flexk.jsonl \
    --k2 sentence_retrieval_k2.jsonl \
    --k3 sentence_retrieval_k3.jsonl \
    --k4 sentence_retrieval_k4.jsonl \
    --k5 sentence_retrieval_k5.jsonl
```

#### Evaluate
```sh
python sentence_retrieval/evaluate.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --sentence-retrieval sentence_retrieval.jsonl
```

-------------------------------------------------------

## NLI

#### Oracle
```sh
python nli/oracle.py \
    --dataset data/dataset.test.jsonl \
    --sentence-retrieval sentence_retrieval.jsonl \
    --output nli.jsonl
```

#### Transformers
```sh
python nli/transformer.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --sentence-retrieval sentence_retrieval.jsonl \
    --model models/nli_scibert_scifact \
    --output nli.jsonl
```


#### Evaluate
```sh
python nli/evaluate.py \
    --corpus data/corpus.jsonl \
    --dataset data/dataset.test.jsonl \
    --nli nli.jsonl
```

-------------------------------------------------------

## Pipeline

#### Evaluate
```sh
python pipeline/evaluate.py \
    --dataset data/dataset.test.jsonl \
    --sentence-retrieval sentence_retrieval.jsonl \
    --nli nli.jsonl
```
