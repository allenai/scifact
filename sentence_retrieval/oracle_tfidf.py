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
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
output = jsonlines.open(args.output, 'w')

for data in dataset:
    evidence = {}

    if data['evidence']:
        for doc_id, evidence_set in data['evidence'].items():
            evidence[doc_id] = [s for es in evidence_set for s in es['sentences']]
    else:
        doc_id = data['cited_doc_ids'][0]
        sentences = corpus[int(doc_id)]['abstract']
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
