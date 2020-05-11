# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) by David Wadden, Kyle Lo, Lucy Lu Wang, Shanchuan Lin, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.

- Project [website](https://scifact.apps.allenai.org).

## Data

To download the data from Amazon S3, invoke
```bash
bash script/download-data.sh
```
The data will be downloaded and stored in the `data` directory.

Or, [click here](https://ai2-s2-scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz) to download the tarball.

The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The labels are removed for the test set. The corpus of evidence documents is `corpus.jsonl`, one evidence document per line. See [data.md](data.md) for descriptions of the schemas for each file type.

## Modeling baselines

The pre-trained models described in the preprint are available for download from Amazon S3. For more information on downloading the models and using them to make predictions, see [baselines.md](baselines.md).

## Evaluation scripts

We will add an evaluation script shortly.

## Training scripts

See [training.md](training.md).

## Contact

Email: `davidw@allenai.org`.
