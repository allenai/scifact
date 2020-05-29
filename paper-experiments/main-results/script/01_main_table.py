"""
Pull together results from dataset and pipeline ablations.
"""

import pandas as pd

from pipeline_components import pipeline_components
from dataset_variants import dataset_variants


pipelined = pipeline_components()
datasets = dataset_variants()

res = pd.concat([pipelined[:-1], datasets])

with pd.option_context("max_colwidth", 1000):
    to_latex = res.to_latex(escape=False, na_rep="", float_format="%0.1f")
    with open(f"results/metrics/main-table.tex", "w") as f:
        print(to_latex, file=f)
