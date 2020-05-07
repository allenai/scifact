import argparse

from lib.data import GoldDataset, PredictedDataset
from lib import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
args = parser.parse_args()

data = GoldDataset(args.corpus, args.dataset)
predictions = PredictedDataset(
    data,
    args.rationale_selection,
    args.label_prediction
)

res = metrics.compute_metrics(predictions)
print(res)
