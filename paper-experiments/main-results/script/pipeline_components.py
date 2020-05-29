import numpy as np
import pandas as pd

from lib import util, release
from lib.report import Report
from lib.metrics_full_pipeline import compute_metrics
from lib.release import Label
from collections import Counter

from itertools import product


orders = [
    ("oracle", "oracle", "system"),
    ("oracle", "system", "oracle"),
    ("oracle", "system", "system"),
    ("system", "oracle", "system"),
    ("system", "system", "oracle"),
    ("system", "system", "system")
]

names = ["oos",
         "oso",
         "oss",
         "sos",
         "sso",
         "sss"]

assert len(set(orders)) == len(orders) == 6


def format_score(score, decode_method):
    score["decode_method"] = decode_method
    res = score.reset_index().melt(
        id_vars=["index", "decode_method"], value_vars=["coarse", "fine"])
    res.columns = ["prf", "decode_method", "metric", "value"]
    return res.set_index(["metric", "decode_method", "prf"])


def make_rowname(modules):
    lookup = {"oracle": "Oracle", "system": "\sysname"}
    components = [lookup[module] for module in modules]
    name = " $\rightarrow$ ".join(components)
    return name


def convert_to_tbl(row, modules):
    filler = pd.Series([""], index=[("filler", "filler")])
    filler.name = "value"

    # Do fixed-k for coarse, and flex-k for fine.
    to_tbl = pd.concat([row["flex"].loc["fine"],
                        filler,
                        row["flex"].loc["coarse"]])
    to_tbl = to_tbl["value"] * 100

    return to_tbl.to_frame().T


def pipeline_components():
    arxiv_dir = "../arxiv-2020-04-11"
    data = release.Release(fold_dir=f"{arxiv_dir}/results/corpus/folds", corpus_file=f"{arxiv_dir}/results/corpus/corpus.jsonl",
                           admin_file=f"{arxiv_dir}/results/corpus/evidence-admin.jsonl")

    decode_methods = ["flex"]

    final_table = []
    for order in orders:
        row = {}
        for decode_method in decode_methods:
            abstract, rationale, entailment = order
            rationale_name = f"{rationale}_{decode_method}" if "oracle" not in rationale else rationale
            dirname = f"{arxiv_dir}/data/predictions/results_combinations/{abstract}.{rationale_name}.{entailment}"
            rationale_file = f"{dirname}/sentence_retrieval.jsonl"
            entailment_file = f"{dirname}/nli.jsonl"
            preds = release.PredictedDataset(
                data, rationale_file, entailment_file)
            score = compute_metrics(preds)
            import ipdb; ipdb.set_trace()
            score = format_score(score, decode_method)
            row[decode_method] = score

        to_tbl = convert_to_tbl(row, order)
        final_table.append(to_tbl)

    final_table = pd.concat(final_table)
    final_table.index = names

    return final_table
