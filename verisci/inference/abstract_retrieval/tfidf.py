import argparse
import jsonlines
import numpy as np
from statistics import mean, median
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--min-gram', type=int, required=True)
parser.add_argument('--max-gram', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = list(jsonlines.open(args.corpus))
dataset = list(jsonlines.open(args.dataset))
output = jsonlines.open(args.output, 'w')
k = args.k

vectorizer = TfidfVectorizer(stop_words='english',
                             ngram_range=(args.min_gram, args.max_gram))

doc_vectors = vectorizer.fit_transform([doc['title'] + ' '.join(doc['abstract'])
                                        for doc in corpus])

doc_ranks = []

for data in dataset:
    claim = data['claim']
    claim_vector = vectorizer.transform([claim]).todense()
    doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze()
    doc_indices_rank = doc_scores.argsort()[::-1].tolist()
    doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]

    for gold_doc_id in data['evidence'].keys():
        rank = doc_id_rank.index(int(gold_doc_id))
        doc_ranks.append(rank)

    output.write({
        'claim_id': data['id'],
        'doc_ids': doc_id_rank[:k]
    })

print(f'Mid reciprocal rank: {median(doc_ranks)}')
print(f'Avg reciprocal rank: {mean(doc_ranks)}')
print(f'Min reciprocal rank: {min(doc_ranks)}')
print(f'Max reciprocal rank: {max(doc_ranks)}')
