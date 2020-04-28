"""
Evaluates document retrieval.
"""

import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--document-retrieval', type=str, required=True)
args = parser.parse_args()

dataset = {data['id']: data for data in jsonlines.open(args.dataset)}

hit_one = 0
hit_all = 0
total = 0
for retrieval in jsonlines.open(args.document_retrieval):
    total += 1
    data = dataset[retrieval['claim_id']]

    pred_doc_ids = set(retrieval['doc_ids'])
    true_doc_ids = set(map(int, data['evidence'].keys()))

    if pred_doc_ids.intersection(true_doc_ids) or not true_doc_ids:
        hit_one += 1
    if pred_doc_ids.issuperset(true_doc_ids):
        hit_all += 1

print(f'Hit one: {round(hit_one / total, 4)}')
print(f'Hit all: {round(hit_all / total, 4)}')
