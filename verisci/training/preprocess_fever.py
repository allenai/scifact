"""
This script converts Fever dataset to a our format so that it can pass
in to our training script.

You will need to manually download the dataset and wiki dump.

1. Download Fever dataset:
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl

2. Download Wikipedia Dump and unzip it manually.
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
"""

import argparse
import jsonlines
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--wiki-folder', type=str, required=True, help='Path to wiki-pages folder')
parser.add_argument('--input', type=str, required=True, help='Path to train.jsonl, paper_dev.jsonl or paper_test.jsonl')
parser.add_argument('--output', type=str, required=True, help='Path to output jsonl, i.e fever_train.jsonl')
args = parser.parse_args()

wiki_dict = {}

print('Loading wiki dump into memory')
for filename in tqdm(sorted(os.listdir(args.wiki_folder))):
    for wiki_doc in jsonlines.open(os.path.join(args.wiki_folder, filename)):
        wiki_id = wiki_doc['id']
        wiki_sentences = wiki_doc['lines']\
          .replace('-LRB-', '(')\
          .replace('-RRB-', ')')\
          .replace('-LSB-', '(')\
          .replace('-RSB-', ')')\
          .split('\n')
        wiki_sentences = [s[s.find('\t')+1:].replace('\t', ' ') for s in wiki_sentences]
        wiki_dict[wiki_id] = wiki_sentences


print('Converting dataset')
claims = list(jsonlines.open(args.input))
output = jsonlines.open(args.output, 'w')

for claim_doc in tqdm(claims):
    claim_id = claim_doc['id']
    claim = claim_doc['claim']
    label = claim_doc['label']

    # Merge wiki documents to sentences
    wiki_ids = {e[2] for es in claim_doc['evidence'] for e in es if e[2]}
    sentences = []
    sentence_mapping = {}

    for wiki_id in wiki_ids:
        wiki_sentences = wiki_dict.get(wiki_id)
        if wiki_sentences:
            for i, wiki_sentence in enumerate(wiki_sentences):
                if wiki_sentence:  # filter out empty line
                    sentence_mapping[(wiki_id, i)] = len(sentences)
                    sentences.append(wiki_sentence)
        else:
            label = 'NOT ENOUGH INFO'

    # Process evidence set
    evidence_sets = []
    for es in claim_doc['evidence']:
        evidence_sentences = [sentence_mapping[(e[2], e[3])] for e in es if e[2] in wiki_dict.keys()]
        evidence_sentences = list(sorted(evidence_sentences))
        if evidence_sentences and evidence_sentences not in evidence_sets:
            evidence_sets.append(evidence_sentences)

    output.write({
        'id': claim_id,
        'claim': claim,
        'label': label,
        'sentences': sentences,
        'evidence_sets': evidence_sets
    })
