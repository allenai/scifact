import argparse
import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=True)
parser.add_argument('--min-gram', type=int, required=True)
parser.add_argument('--max-gram', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
abstract_retrieval = jsonlines.open(args.abstract_retrieval)
output = jsonlines.open(args.output, 'w')

for data, retrieval in zip(dataset, abstract_retrieval):
    assert data['id'] == retrieval['claim_id']

    evidence = {}
    for doc_id in retrieval['doc_ids']:
        doc = corpus[doc_id]
        vectorizer = TfidfVectorizer(stop_words='english',
                                     ngram_range=(args.min_gram, args.max_gram))
        sentence_vectors = vectorizer.fit_transform(doc['abstract'])
        claim_vector = vectorizer.transform([data['claim']]).todense()
        sentence_scores = np.asarray(sentence_vectors @ claim_vector.T).squeeze()
        top_sentence_indices = sentence_scores.argsort()[-args.k:][::-1].tolist()
        top_sentence_indices.sort()
        evidence[doc_id] = top_sentence_indices

    output.write({
        'claim_id': retrieval['claim_id'],
        'evidence': evidence
    })
