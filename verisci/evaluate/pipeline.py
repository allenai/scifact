import argparse

from lib.data import GoldDataset, PredictedDataset
from lib import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--prediction', type=str, required=True)
args = parser.parse_args()

data = GoldDataset(args.corpus, args.dataset)
predictions = PredictedDataset(data, args.prediction)

res = metrics.compute_metrics(predictions)
print(res)
