import argparse
import jsonlines


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall) if precision + recall else 0


def metrics(trues, preds):
    true_supports = [x for x in trues if x[2] == 'SUPPORT']
    pred_supports = [x for x in preds if x[2] == 'SUPPORT']
    support_hits = 0
    for p in pred_supports:
        for t in true_supports:
            if p[0] == t[0] and p[1] == t[1] and any(set(p[3]).issuperset(e) for e in t[3]):
                support_hits += 1
                continue
    precision_support = support_hits / len(pred_supports) if len(pred_supports) else 0
    recall_support = support_hits / len(true_supports) if len(true_supports) else 0
    f1_support = f1_score(precision_support, recall_support)

    true_contradicts = [x for x in trues if x[2] == 'CONTRADICT']
    pred_contradicts = [x for x in preds if x[2] == 'CONTRADICT']
    contradict_hits = 0
    for p in pred_contradicts:
        for t in true_contradicts:
            if p[0] == t[0] and p[1] == t[1] and any(set(p[3]).issuperset(e) for e in t[3]):
                contradict_hits += 1
                continue
    precision_contradict = contradict_hits / len(pred_contradicts) if len(pred_contradicts) else 0
    recall_contradict = contradict_hits / len(true_contradicts) if len(true_contradicts) else 0
    f1_contradict = f1_score(precision_contradict, recall_contradict)

    macro_f1 = (f1_support + f1_contradict) / 2

    return {
        'macro_f1': round(macro_f1, 4),
        'f1_support': round(f1_support, 4),
        'precision_support': round(precision_support, 4),
        'recall_support': round(recall_support, 4),
        'f1_contradict': round(f1_contradict, 4),
        'precision_contradict': round(precision_contradict, 4),
        'recall_contradict': round(recall_contradict, 4)
    }


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--sentence-retrieval', type=str, required=True)
parser.add_argument('--nli', type=str, required=True)
args = parser.parse_args()

dataset = {data['id']: data for data in jsonlines.open(args.dataset)}
sentence_retrievals = list(jsonlines.open(args.sentence_retrieval))
nli_results = jsonlines.open(args.nli)

trues = []
for data in dataset.values():
    if data['evidence']:
        claim_id = data['id']
        for doc_id, evidence_sets in data['evidence'].items():
            evidence_sets = [es['sentences'] for es in evidence_sets]
            trues.append((claim_id, int(doc_id), data['label'], evidence_sets))

document_preds = []
for sentence_retrieval in sentence_retrievals:
    claim_id = sentence_retrieval['claim_id']
    data = dataset[claim_id]
    for doc_id in sentence_retrieval['evidence'].keys():
        if data['evidence'].get(doc_id):
            gold_evidence = [s for es in data['evidence'][doc_id] for s in es['sentences']]
            document_preds.append(
                (claim_id, int(doc_id), data['label'], gold_evidence))

retrieval_preds = []
for sentence_retrieval in sentence_retrievals:
    claim_id = sentence_retrieval['claim_id']
    data = dataset[claim_id]
    for doc_id, evidence in sentence_retrieval['evidence'].items():
        if data['evidence'].get(doc_id):
            retrieval_preds.append((claim_id, int(doc_id), data['label'], evidence))


pipeline_preds = []
for nli, sentence_retrieval in zip(nli_results, sentence_retrievals):
    assert nli['claim_id'] == sentence_retrieval['claim_id']
    for doc_id, label in nli['labels'].items():
        if label['label'] != 'NOT_ENOUGH_INFO':
            pipeline_preds.append((nli['claim_id'], int(doc_id), label['label'], sentence_retrieval['evidence'][doc_id]))

print(f'Document Retrieval: {metrics(trues, document_preds)}')
print(f'Sentence Retrieval: {metrics(trues, retrieval_preds)}')
print(f'Full pipeline:      {metrics(trues, pipeline_preds)}')



