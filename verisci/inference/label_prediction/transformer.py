import argparse
import jsonlines

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--mode', type=str, default='claim_and_rationale', choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

print(args.mode)

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).eval().to(device)

LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']

def encode(sentences, claims):
    text = {
        "claim_and_rationale": list(zip(sentences, claims)),
        "only_claim": claims,
        "only_rationale": sentences
    }[args.mode]
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            pad_to_max_length=True,
            truncation_strategy='only_first',
            return_tensors='pt'
        )
    encoded_dict = {key: tensor.to(device)
                  for key, tensor in encoded_dict.items()}
    return encoded_dict


with torch.no_grad():
    for data, selection in tqdm(list(zip(dataset, rationale_selection))):
        assert data['id'] == selection['claim_id']

        claim = data['claim']
        results = {}
        for doc_id, indices in selection['evidence'].items():
            if not indices:
                results[doc_id] = {'label': 'NOT_ENOUGH_INFO', 'confidence': 1}
            else:
                evidence = ' '.join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                encoded_dict = encode([evidence], [claim])
                label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                label_index = label_scores.argmax().item()
                label_confidence = label_scores[label_index].item()
                results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4)}
        output.write({
            'claim_id': data['id'],
            'labels': results
        })
