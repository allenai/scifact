"""
Performs sentence retrieval with oracle on SUPPORT and CONTRADICT claims,
and tfidf on NOT_ENOUGH_INFO claims
"""

import argparse
import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
abstract_retrieval = jsonlines.open(args.abstract_retrieval)
dataset = jsonlines.open(args.dataset)
output = jsonlines.open(args.output, 'w')

for data, retrieval in zip(dataset, abstract_retrieval):
    assert data['id'] == retrieval['claim_id']

    evidence = {}

    for doc_id in retrieval['doc_ids']:
        if data['evidence'].get(str(doc_id)):
            evidence[doc_id] = [s for es in data['evidence'][str(doc_id)] for s in es['sentences']]
        else:
            sentences = corpus[doc_id]['abstract']
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            claim_vector = vectorizer.transform([data['claim']]).todense()
            sentence_scores = np.asarray(sentence_vectors @ claim_vector.T).squeeze()
            top_sentence_indices = sentence_scores.argsort()[-2:][::-1].tolist()
            top_sentence_indices.sort()
            evidence[doc_id] = top_sentence_indices

    output.write({
        'claim_id': data['id'],
        'evidence': evidence
    })
