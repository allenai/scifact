# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.

- Project [website](https://scifact.apps.allenai.org).

## Table of contents
- [Dependencies](#dependencies)
- [Run models for paper metrics](#run-models-for-paper-metrics)
- [Download data set](#download-data-set)
- [Download pre-trained models](#download-pre-trained-models)
- [Training scripts](#training-scripts)
- [Leaderboard](#leaderboard)
- [Contact](#contact)


## Dependencies

We recommend you create an anaconda environment:
```bash
conda create --name scifact python=3.7 conda-build
```
Then, from the `scifact` project root, run
```
conda develop .
```
which will add the scifact code to your `PYTHONPATH`.

Then, install Python requirements:
```
pip install -r requirements.txt
```

## Run models for paper metrics

We provide scripts let you easily run our models and re-create the metrics published in paper. The script will automatically download the dataset and pre-trained models. You should be able to reproduce our dev set results from the paper by following these instructions (we are not releasing test set labels at this point). Please post an issue if you're unable to do this.

To recreate table 3 rationale selection metrics:
```bash
./script/rationale-selection.sh [bert-variant] [training-dataset] [dataset]
```
To recreate table 3 label prediction metrics:
```bash
./script/label-prediction.sh [bert-variant] [training-dataset] [dataset]
```
- `[bert-variant]` options: `roberta_large`, `roberta_base`, `scibert`, `biomed_roberta_base`
- `[training-dataset]` options: `scifact`, `scifact_only_claim`, `scifact_only_rationale`, `fever_scifact`, `fever`, `snopes`
- `[dataset]` options: `dev`, `test`

To recreate table 7:
```bash
./script/pipeline.sh [retrieval] [model] [dataset]
```
- `[retrieval]` options: `oracle`, `open`
- `[model]` options: `oracle-rationle`, `zero-shot`, `verisci`
- `[dataset]` options: `dev`, `test`

Some numbers may be off by 0.1 F1, since in Table 7 report the mean over 10,000 bootstrap-resampled dev sets and here we just report the result on the full dev set.

## Download data set

Download with script: The data will be downloaded and stored in the `data` directory.
```bash
./script/download-data.sh
```
Or, [click here](https://scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz) to download the tarball.

The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The labels are removed for the test set. The corpus of evidence documents is `corpus.jsonl`, one evidence document per line.

See [data.md](doc/data.md) for descriptions of the schemas for each file type.

## Download pre-trained models

All "BERT-to-BERT"-style models as described in the paper are stored in a public AWS S3 bucket. You can download the models models using the script:
```bash
./script/download-model.sh [model-component] [bert-variant] [training-dataset]
```
- `[model-component]` options: `rationale`, `label`
- `[bert-variant]` options: `roberta_large`, `roberta_base`, `scibert`, `biomed_roberta_base`
- `[training-dataset]` options: `scifact`, `scifact_only_claim`, `scifact_only_rationale`, `fever_scifact`, `fever`, `snopes`

The script checks to make sure the downloaded model doesn't already exist before starting new downloads.

The best-performing pipeline reported in [paper](https://arxiv.org/abs/2004.14974) uses:
- `rationale`: `roberta_large` + `scifact`
- `label`: `roberta_large` + `fever_scifact`

For `fever` and `fever_scifact`, there are models available for all 4 BERT variants. For `snopes`, only `roberta_large` is available for download (but you can train your own model).

After downloading the pretrained-model, you can follow instruction [model.md](doc/model.md) to run individual model components.

## Training scripts

See [training.md](training.md).

## Leaderboard

In progress. For now, feel free to email test-set predictions to the contact below.

## Contact

Email: `davidw@allenai.org`.
