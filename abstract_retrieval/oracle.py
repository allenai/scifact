import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--include-nei', action='store_true')
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
output = jsonlines.open(args.output, 'w')

for data in dataset:
    doc_ids = list(map(int, data['evidence'].keys()))
    if not doc_ids and args.include_nei:
        doc_ids = [data['cited_doc_ids'][0]]

    output.write({
        'claim_id': data['id'],
        'doc_ids': doc_ids
    })
