# SciFact

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.

⬇️[**Download the dataset here**](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz).

🏆[**Participate in SCIVER shared task here**](https://sdproc.org/2021/sharedtasks.html#sciver).

📈[**Check out the leaderboard here**](https://leaderboard.allenai.org/scifact/).

You can also check out our COVID-19 claim verification [demo](https://scifact.apps.allenai.org). For a heavier-weight COVID claim verifier, see the section on [verifying COVID-19 claims](#verify-claims-about-covid-19).


**Update (Dec 2020)**: SciFact will be used for the [SciVer](https://sdproc.org/2021/sharedtasks.html#sciver) shared task to be featured at the SDP workshop at NAACL 2021.  Registration is open!


**Update (Dec 2020)**: Claim / citation context data now available to train claim generation models. See [Claim generation data](#claim-generation-data).

## Table of contents
- [Leaderboard](#leaderboard)
- [Dependencies](#dependencies)
- [Run models for paper metrics](#run-models-for-paper-metrics)
- [Dataset](#dataset)
- [Download pre-trained models](#download-pre-trained-models)
- [Training scripts](#training-scripts)
- [Verify claims about COVID-19](#verify-claims-about-covid-19)
- [Citation](#citation)
- [Contact](#contact)


## Leaderboard

**UPDATE (Jan 2021)**: We now have an official [AI2 leaderboard](https://leaderboard.allenai.org/scifact/) with automated evaluation! For information on the submission file format and evaluation metrics, see [evaluation.md](doc/evaluation.md). Or, check out the [getting started](https://leaderboard.allenai.org/scifact/submissions/get-started) page on the leaderboard.


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

To recreate Table 3 rationale selection metrics:
```bash
./script/rationale-selection.sh [bert-variant] [training-dataset] [dataset]
```
To recreate Table 3 label prediction metrics:
```bash
./script/label-prediction.sh [bert-variant] [training-dataset] [dataset]
```
- `[bert-variant]` options: `roberta_large`, `roberta_base`, `scibert`, `biomed_roberta_base`
- `[training-dataset]` options: `scifact`, `scifact_only_claim`, `scifact_only_rationale`, `fever_scifact`, `fever`, `snopes`
- `[dataset]` options: `dev`, `test`

To recreate Table 4:
```bash
./script/pipeline.sh [retrieval] [model] [dataset]
```
- `[retrieval]` options: `oracle`, `open`
- `[model]` options: `oracle-rationale`, `zero-shot`, `verisci`
- `[dataset]` options: `dev`, `test`


## Dataset

Download with script: The data will be downloaded and stored in the `data` directory.
```bash
./script/download-data.sh
```
Or, [click here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz) to download the tarball.

The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The claim and dev sets contain labels, while the test set is unlabeled. For test set evaluation, submit to the [leaderboard](https://leaderboard.allenai.org/scifact)! The corpus of evidence documents is `corpus.jsonl`, one evidence document per line.

Due to the relatively small size of the dataset, we also provide a 5-fold cross-validation split that may be useful for model development. After unzipping the tarball, the data will organized like this:

```
data
| corpus.jsonl
| claims_train.jsonl
| claims_dev.jsonl
| claims_test.jsonl
| cross_validation
  | fold_1
    | claims_train_1.jsonl
    | claims_dev_1.jsonl
  ...
  | fold_5
    | claims_train_5.jsonl
    | claims_dev_5.jsonl
```

See [data.md](doc/data.md) for descriptions of the schemas for each file type.


### Claim generation data

We also make available the collection of claims together with the documents and citation contexts they are based on. We hope that these data will facilitate the training of "claim generation" models that can summarize a citation context into atomic claims. Click [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl) to download the file, or enter

```bash
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl -P data
```

For more information on the data, see [claims-with-citances.md](doc/claims-with-citances.md)


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

See [training.md](doc/training.md).

## Verify claims about COVID-19

While the project [website](https://scifact.apps.allenai.org) features a COVID-19 fact-checking demo, it is not configurable and uses a "light-weight" version of VeriSci based on [DistilBERT](https://arxiv.org/abs/1910.01108). We provide a more configurable fact-checking script that uses the full model. Like the web demo, it uses [covidex](https://covidex.ai) for document retrieval.  Usage is as follows:

```shell
python script/verify_covid.py [claim-text] [report-file] [optional-arguments].
```

For a description of the optional arguments, run `python script/verify_covid.py -h`. The script generates either a `pdf` or `markdown` report. The `pdf` version requires [pandoc](https://pandoc.org) and [wkhtmltopdf](https://wkhtmltopdf.org), both of which can be installed with `conda`. A usage example might be:

```shell
python script/verify_covid.py \
  "Coronavirus droplets can remain airborne for hours" \
  results/covid-report \
  --n_documents=100 \
  --rationale_selection_method=threshold \
  --rationale_threshold=0.2 \
  --verbose \
  --full_abstract
```

The 36 claims COVID claims mentions in the paper can be found at [covid/claims.txt](covid/claims.txt).

## Citation

```bibtex
@inproceedings{Wadden2020FactOF,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  booktitle={EMNLP},
  year={2020},
}
```

## Contact

Email: `davidw@allenai.org`.
