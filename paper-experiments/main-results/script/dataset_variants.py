import numpy as np
import pandas as pd

from lib import util, release
from lib.report import Report
from lib.metrics_full_pipeline import compute_metrics
from lib.release import Label
from collections import Counter

from itertools import product


def format_score(score, decode_method):
    score["decode_method"] = decode_method
    res = score.reset_index().melt(
        id_vars=["index", "decode_method"], value_vars=["coarse", "fine"])
    res.columns = ["prf", "decode_method", "metric", "value"]
    return res.set_index(["metric", "decode_method", "prf"])


def convert_to_tbl(row, dataset):
    lookup = {"fever": "\\fever", "snopes": "\snopes", "verisci": "\sysname"}

    filler = pd.Series([np.nan], index=[("filler", "filler")])
    filler.name = "value"

    # to_tbl = pd.concat([row["k3"].loc["fine"], row["flex"].loc["fine"],
    #                     filler,
    #                     row["k3"].loc["coarse"], row["flex"].loc["coarse"]])

    tbl = {
        "k3": pd.concat([row["k3"].loc["fine"], filler, row["k3"].loc["coarse"]]),
        "flex": pd.concat([row["flex"].loc["fine"], filler, row["flex"].loc["coarse"]])
    }

    for k, to_tbl in tbl.items():
        to_tbl = to_tbl[["value"]].T
        to_tbl.index = [lookup[dataset]]
        tbl[k] = to_tbl

    return tbl


def dataset_variants():
    # TODO(dwadden) refactor this after paper is done.
    arxiv_dir = "../arxiv-2020-04-11"

    data = release.Release(fold_dir=f"{arxiv_dir}/results/corpus/folds", corpus_file=f"{arxiv_dir}/results/corpus/corpus.jsonl",
                           admin_file=f"{arxiv_dir}/results/corpus/evidence-admin.jsonl")

    decode_methods = ["k3", "flex"]

    datasets = ["fever", "snopes", "verisci"]

    final_tables = {"k3": [], "flex": []}
    for dataset in datasets:
        row = {}
        for decode_method in decode_methods:
            dirname = f"{arxiv_dir}/data/predictions/results_pipeline_final/{dataset}_{decode_method}"
            rationale_file = f"{dirname}/sentence_retrieval.jsonl"
            entailment_file = f"{dirname}/nli.jsonl"
            preds = release.PredictedDataset(
                data, rationale_file, entailment_file)
            score = compute_metrics(preds)
            score = format_score(score, decode_method)
            row[decode_method] = score

        to_tbl = convert_to_tbl(row, dataset)
        for k, v in to_tbl.items():
            final_tables[k].append(v)

    ix = ["\\fever",
          "\\snopes",
          "\sysname"]

    final_tables = {k: pd.concat(final_table) *
                    100 for k, final_table in final_tables.items()}
    for k in final_tables:
        final_tables[k].index = ix

    return final_tables["flex"]
