import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--sentence-retrieval', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
sentence_retrieval = jsonlines.open(args.sentence_retrieval)
output = jsonlines.open(args.output, 'w')

for data, retrieval in zip(dataset, sentence_retrieval):
    assert data['id'] == retrieval['claim_id']

    labels = {}
    for doc_id, pred_evidence in retrieval['evidence'].items():
        if data['evidence'].get(doc_id):
            true_evidence_set = [es['sentences'] for es in data['evidence'][doc_id]]
            if any(set(es).issubset(pred_evidence) for es in true_evidence_set):
                labels[doc_id] = {'label': data['label'], 'confidence': 1}
                continue
        labels[doc_id] = {'label': 'NOT_ENOUGH_INFO', 'confidence': 1}

    output.write({
        'claim_id': data['id'],
        'labels': labels
    })
