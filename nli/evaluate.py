"""
Evaluates NLI results
"""

import argparse
import jsonlines

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--nli', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
nli_results = jsonlines.open(args.nli)

pred_labels = []
true_labels = []

LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

hit_label = set()
total = set()

for data, nli in zip(dataset, nli_results):
    assert data['id'] == nli['claim_id']

    if args.filter:
        nli['labels'] = {doc_id: pred for doc_id, pred in nli['labels'].items()
                         if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
    if not nli['labels']:
        continue

    claim_id = data['id']
    total.add(claim_id)
    for doc_id, pred in nli['labels'].items():
        pred_label = pred['label']
        true_label = data['label'] if data['evidence'].get(doc_id) else 'NOT_ENOUGH_INFO'
        pred_labels.append(LABELS[pred_label])
        true_labels.append(LABELS[true_label])
        if pred_label == true_label:
            hit_label.add(claim_id)

print(f'Hit Label          {round(len(hit_label) / len(total), 4)}')
print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
print(f'F1:                {f1_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Precision:         {precision_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Recall:            {recall_score(true_labels, pred_labels, average=None).round(4)}')
print()
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
