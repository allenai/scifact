"""
Confusion matrix on the pipeline result
"""

import argparse
import jsonlines
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--sentence-retrieval', type=str, required=True)
parser.add_argument('--nli', type=str, required=True)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
sentence_retrieval = jsonlines.open(args.sentence_retrieval)
nli_results = jsonlines.open(args.nli)

LABEL_MAP = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

true = []
pred = []

for data, retrieval, nli in zip(dataset, sentence_retrieval, nli_results):
    assert data['id'] == retrieval['claim_id'] == nli['claim_id']
    for doc_id in nli['labels']:
        true_label = data['label'] if data['evidence'].get(doc_id) and any(set(retrieval['evidence'][doc_id]).issuperset(es['sentences']) for es in data['evidence'][doc_id]) else 'NOT_ENOUGH_INFO'
        pred_label = nli['labels'][doc_id]['label']
        true.append(LABEL_MAP[true_label])
        pred.append(LABEL_MAP[pred_label])

print(confusion_matrix(true, pred))
