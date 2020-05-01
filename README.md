# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974).

- Project [website](https://github.com/allenai/scifact-demo).

## Data

The data are located in the `data` subdirectory of this repository. The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The labels are removed for the test set. The corpus of evidence documents is `corpus.jsonl`, one evidence document per line. See [data.md](data.md) for descriptions of the schemas for each file type.

## Modeling baselines

Pre-trained models are available. For more information on downloading the models and using them to make predictions, see [baselines.md](baselines.md).

## Evaluation scripts

We will add an evaluation script shortly.
