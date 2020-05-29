import argparse
import jsonlines

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
label_prediction = jsonlines.open(args.label_prediction)


LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

n_total = 0
n_correct = 0

for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if not data["evidence"]:
        gold_label = "NOT_ENOUGH_INFO"
    else:
        gold_labels = set()
        for entry in data["evidence"].values():
            for rat in entry:
                gold_labels.add(rat["label"])
        assert len(gold_labels) == 1
        gold_label = next(iter(gold_labels))

    pred_labels = [x["label"] for x in prediction["labels"].values()]
    n_total += len(pred_labels)
    correct = [x for x in pred_labels if x == gold_label]
    n_correct += len(correct)

print(n_correct / n_total)
