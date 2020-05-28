import argparse
import jsonlines
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)

hit_one = set()
hit_set = set()
total = set()

pred = set()
true = set()

for data, retrieval in zip(dataset, rationale_selection):
    assert data['id'] == retrieval['claim_id']

    if args.filter is not None:
        retrieval['evidence'] = {doc_id: evidence for doc_id, evidence in retrieval['evidence'].items()
                                 if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
        data['evidence'] = {doc_id: evidence for doc_id, evidence in data['evidence'].items()
                            if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}

    if not retrieval['evidence']:
        continue

    claim_id = retrieval['claim_id']
    total.add(claim_id)

    for doc_id, pred_sentences in retrieval['evidence'].items():
        pred_sentences = set(pred_sentences)
        true_evidence_sets = data['evidence'].get(doc_id) or []

        if any(pred_sentences.intersection(s['sentences']) for s in true_evidence_sets):
            hit_one.add(claim_id)
        if any(pred_sentences.issuperset(s['sentences']) for s in true_evidence_sets):
            hit_set.add(claim_id)

    for doc_id, pred_sentences in retrieval['evidence'].items():
        pred.update([(claim_id, doc_id, s) for s in pred_sentences])

    for doc_id, true_evidence_sets in data['evidence'].items():
        true_sentences = {s for es in true_evidence_sets for s in es['sentences']}
        true.update([(claim_id, doc_id, s) for s in true_sentences])

ys = list(pred.union(true))
yt = [y in true for y in ys]
yp = [y in pred for y in ys]

print(f'Hit one:   {round(len(hit_one) / len(total), 4)}')
print(f'Hit set:   {round(len(hit_set) / len(total), 4)}')
print(f'F1:        {round(f1_score(yt, yp), 4)}')
print(f'Precision: {round(precision_score(yt, yp), 4)}')
print(f'Recall:    {round(recall_score(yt, yp), 4)}')
