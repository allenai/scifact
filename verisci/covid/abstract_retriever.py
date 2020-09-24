import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import spacy


class AbstractRetriever:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_sm")

    def __call__(self, claim: str, k=20):
        print("Retrieving abstracts.")
        req = Request(url=f'https://covidex.ai/api/search?vertical=cord19&query={quote(claim)}',
                      headers={'User-Agent': 'SciFact'})
        res = json.loads(urlopen(req).read())

        # Replace the DOI URL from covidex with the S2 URL, which also takes a
        # DPI.

        out = []
        for data in res['response']:
            if data['abstract']:
                new_url = data['url']                                                 # str
                if 'doi.org' in new_url:
                    new_url = new_url.replace('doi.org', 'api.semanticscholar.org')
                elif 'www.ncbi.nlm.nih.gov/pubmed/' in new_url:
                    new_url = new_url.replace('www.ncbi.nlm.nih.gov/pubmed/', 'api.semanticscholar.org/pmid:')
                elif 'arxiv.org' in new_url:
                    new_url = new_url.replace('.pdf', '')
                    for i in range(1, 6):
                        new_url = new_url.replace(f'v{i}', '')
                    new_url = new_url.replace('arxiv.org/pdf/', 'api.semanticscholar.org/arxiv:')
                blob = {
                    'id': data['id'],                                                 # str
                    'title': data['title'],                                           # str
                    'abstract': self._sentencize(data["abstract"]),                   # List[str]
                    'journal': data['journal'],                                       # str
                    'url': new_url,
                    'authors': data['authors'],                                       # List[str]
                }
                out.append(blob)
        return out[:k]

    def _sentencize(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
