"""
Performs sentence retrieval with oracle
"""

import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--document-retrieval', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
document_retrieval = jsonlines.open(args.document_retrieval)
output = jsonlines.open(args.output, 'w')

for data, retrieval in zip(dataset, document_retrieval):
    assert data['id'] == retrieval['claim_id']

    evidence = {}
    for doc_id in retrieval['doc_ids']:
        doc_id = str(doc_id)
        if data['evidence'].get(doc_id):
            evidence[doc_id] = [s for es in data['evidence'].get(doc_id) for s in es['sentences']]
        else:
            evidence[doc_id] = []

    output.write({
        'claim_id': data['id'],
        'evidence': evidence
    })
