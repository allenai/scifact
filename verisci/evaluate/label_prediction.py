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

pred_labels = []
true_labels = []

LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if args.filter:
        prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
    if not prediction['labels']:
        continue

    claim_id = data['id']
    for doc_id, pred in prediction['labels'].items():
        pred_label = pred['label']
        true_label = {es['label'] for es in data['evidence'].get(doc_id) or []}
        assert len(true_label) <= 1, 'Currently support only one label per doc'
        true_label = next(iter(true_label)) if true_label else 'NOT_ENOUGH_INFO'
        pred_labels.append(LABELS[pred_label])
        true_labels.append(LABELS[true_label])

print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
print(f'F1:                {f1_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Precision:         {precision_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Recall:            {recall_score(true_labels, pred_labels, average=None).round(4)}')
print()
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
