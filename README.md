# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) by David Wadden, Kyle Lo, Lucy Lu Wang, Shanchuan Lin, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.

- Project [website](https://scifact.apps.allenai.org).

## Table of contents
- [Data](#data)
- [Pre-trained models](#pre-trained-models)
- [Full example](#full-example)
- [Contact](#contact)

## Data

To download the data from Amazon S3, invoke
```bash
bash script/download-data.sh
```
The data will be downloaded and stored in the `data` directory.

Or, [click here](https://ai2-s2-scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz) to download the tarball.

The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The labels are removed for the test set. The corpus of evidence documents is `corpus.jsonl`, one evidence document per line. See [data.md](data.md) for descriptions of the schemas for each file type.

## Pre-trained models

All "BERT-to-BERT"-style models as described in the paper are stored in a public AWS S3 bucket. You can download the models models using the script `script/download-model.sh`. The script usage is as follows:

```bash
bash script/download-model.sh [model-component] [bert-variant] [training-dataset]

# Example usage
bash script/download-model.sh rationale roberta_large scifact
```

The script checks to make sure the downloaded model doesn't already exist before starting new downloads.

There are two `model-component`s:
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

For more details on the models, see [model-details.md](model-details.md).

## Full example

See `script/full-example.sh` for a complete example showing how to download models, make predictions for all three stages of the modeling pipeline, and evaluate the full-pipeline results.

The evaluation metrics are out-of-date; they will updated shortly to reflect the values in Table 1 of the paper.

## Contact

Email: `davidw@allenai.org`.
