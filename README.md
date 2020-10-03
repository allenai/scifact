# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi.

Check out the project [website](https://scifact.apps.allenai.org), which includes a leaderboard and a COVID-19 claim verification demo. For a heavier-weight COVID claim verifier, see the section on [verifying COVID-19 claims](#verify-claims-about-covid-19).

## Table of contents
- [Leaderboard](#leaderboard)
- [Dependencies](#dependencies)
- [Run models for paper metrics](#run-models-for-paper-metrics)
- [Download data set](#download-data-set)
- [Download pre-trained models](#download-pre-trained-models)
- [Training scripts](#training-scripts)
- [Verify claims about COVID-19](#verify-claims-about-covid-19)
- [Citation](#citation)
- [Contact](#contact)


## Leaderboard

The leaderboard is hosted on the [SciFact project page](https://scifact.apps.allenai.org/leaderboard). To submit, follow these instructions:

**If you have a Google account (preferred)**: Fill out this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSf-o6ZBXNCiD35f-CHdLxkHRJbcmEBVsTCJDxN_1X5PuhsJBw/viewform?usp=sf_link). You will be asked to submit a `.jsonl` file containing your test set predictions.

**If you do not have a Google account**: E-mail your prediction file to the [contact](#contact) below. Please provide the following information in the email:

- Submitter name, affiliation, and (optionally) a link to your lab / research group.
- The name of your system, and (optionally) a link to the arXiv preprint where it is presented.

We will compute performance metrics and notify you with results. With your consent, we will also add your model's performance to the leaderboard. For information on the submission file format, see [submissions.md](doc/submissions.md).


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
@article{Wadden2020FactOF,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={David Wadden and Kyle Lo and Lucy Lu Wang and Shanchuan Lin and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.14974}
}
```

## Contact

Email: `davidw@allenai.org`.
