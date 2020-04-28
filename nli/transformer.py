"""
Performs nli with transformers
"""

import argparse
import jsonlines

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--sentence-retrieval', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
sentence_retrieval = jsonlines.open(args.sentence_retrieval)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).eval().to(device)

LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']


def encode(sentences, claims):
    encoded_dict = tokenizer.batch_encode_plus(
      zip(sentences, claims),
      pad_to_max_length=True,
      return_tensors='pt'
    )
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
          zip(sentences, claims),
          max_length=512,
          pad_to_max_length=True,
          truncation_strategy='only_first',
          return_tensors='pt'
        )
    encoded_dict = {key: tensor.to(device)
                  for key, tensor in encoded_dict.items()}
    return encoded_dict


with torch.no_grad():
    for data, retrieval in tqdm(list(zip(dataset, sentence_retrieval))):
        claim = data['claim']
        sentences = [' '.join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                     for doc_id, indices in retrieval['evidence'].items() if indices]
        if not sentences:
            output.write({
                'claim_id': data['id'],
                'labels': {}
            })
        else:
            encoded_dict = encode(sentences, [claim] * len(sentences))
            labels_scores = torch.softmax(model(**encoded_dict)[0], dim=1)
            labels_index = labels_scores.argmax(dim=1).tolist()
            labels_confidence = [labels_scores[r, c].item() for r, c in enumerate(labels_index)]

            # print(labels_index, labels_confidence, labels_scores)

            output.write({
                'claim_id': data['id'],
                'labels': {
                    doc_id: {
                        'label': LABELS[index],
                        'confidence': round(confidence, 4)
                    } for doc_id, index, confidence in zip(retrieval['evidence'].keys(), labels_index, labels_confidence)}
            })
