import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

for data, retrieval in zip(dataset, rationale_selection):
    assert data['id'] == retrieval['claim_id']

    labels = {}
    for doc_id, pred_evidence in retrieval['evidence'].items():
        if data['evidence'].get(doc_id):
            true_evidence_set = [es['sentences'] for es in data['evidence'][doc_id]]
            true_label = {es['label'] for es in data['evidence'][doc_id]}
            assert len(true_label) == 1, 'Currently, label should be the same per claim-doc pair'

            if any(set(es).issubset(pred_evidence) for es in true_evidence_set):
                labels[doc_id] = {'label': next(iter(true_label)), 'confidence': 1}
                continue
        labels[doc_id] = {'label': 'NOT_ENOUGH_INFO', 'confidence': 1}

    output.write({
        'claim_id': data['id'],
        'labels': labels
    })
