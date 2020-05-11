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
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
args = parser.parse_args()

dataset = {data['id']: data for data in jsonlines.open(args.dataset)}
rationale_selection = list(jsonlines.open(args.rationale_selection))
label_prediction = jsonlines.open(args.label_prediction)


def get_gold_label(claim_id: int, doc_id: int):
    labels = {es['label'] for es in dataset[claim_id]['evidence'].get(str(doc_id)) or []}
    if labels:
        return next(iter(labels))
    else:
        return 'NOT_ENOUGH_INFO'

trues = []
for data in dataset.values():
    if data['evidence']:
        claim_id = data['id']
        for doc_id, evidence_sets in data['evidence'].items():
            evidence_sets = [es['sentences'] for es in evidence_sets]
            trues.append((claim_id, int(doc_id), get_gold_label(claim_id, str(doc_id)), evidence_sets))

document_preds = []
for selection in rationale_selection:
    claim_id = selection['claim_id']
    data = dataset[claim_id]
    for doc_id in selection['evidence'].keys():
        if data['evidence'].get(doc_id):
            gold_evidence = [s for es in data['evidence'][doc_id] for s in es['sentences']]
            document_preds.append(
                (claim_id, int(doc_id), get_gold_label(claim_id, str(doc_id)), gold_evidence))

retrieval_preds = []
for selection in rationale_selection:
    claim_id = selection['claim_id']
    data = dataset[claim_id]
    for doc_id, evidence in selection['evidence'].items():
        if data['evidence'].get(doc_id) and any(set(evidence).issuperset(es['sentences']) for es in data['evidence'][doc_id]):
            retrieval_preds.append((claim_id, int(doc_id), get_gold_label(claim_id, str(doc_id)), evidence))

pipeline_preds = []
for prediction, selection in zip(label_prediction, rationale_selection):
    assert prediction['claim_id'] == selection['claim_id']
    for doc_id, label in prediction['labels'].items():
        if label['label'] != 'NOT_ENOUGH_INFO':
            pipeline_preds.append((prediction['claim_id'], int(doc_id), label['label'], selection['evidence'][doc_id]))

print(f'Document Retrieval: {metrics(trues, document_preds)}')
print(f'Sentence Retrieval: {metrics(trues, retrieval_preds)}')
print(f'Full pipeline:      {metrics(trues, pipeline_preds)}')



