"""
Code to represent a dataset release.
"""

from enum import Enum
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple


def load_jsonl(fname):
    return [json.loads(line) for line in open(fname)]


def write_jsonl(data, fname):
    with open(fname, "w") as f:
        for line in data:
            print(json.dumps(line), file=f)


@dataclass(repr=False, frozen=True)
class Document:
    id: str
    title: str
    sentences: Tuple[str]

    def __repr__(self):
        return self.title.upper() + "\n" + "\n".join(["- " + entry for entry in self.sentences])

    def __lt__(self, other):
        return self.title.__lt__(other.title)

    def dump(self):
        res = {"doc_id": self.id,
               "title": self.title,
               "abstract": self.sentences,
               "structured": self.is_structured()}
        return json.dumps(res)


@dataclass(repr=False, frozen=True)
class Corpus:
    """
    A Corpus is just a collection of `Document` objects, with methods to look up
    a single document.
    """
    documents: List[Document]

    def __repr__(self):
        return f"Corpus of {len(self.documents)} documents."

    def __getitem__(self, i):
        "Get document by index in list."
        return self.documents[i]

    def get_document(self, doc_id):
        "Get document by ID."
        res = [x for x in self.documents if x.id == doc_id]
        assert len(res) == 1
        return res[0]


class Label(Enum):
    SUPPORTS = 1
    NEI = 0
    REFUTES = -1


# Classes to read in and handle a release.

class Release:
    """
    Class to represent a full release of the data.
    """
    def __init__(self, corpus_file, data_file):
        self.corpus = self._read_corpus(corpus_file)
        self.examples = self._read_examples(data_file)
    # def __init__(self, fold_dir, corpus_file, admin_file):
    #     self.corpus = self._read_corpus(corpus_file)
    #     self.examples = self._read_examples(fold_dir, admin_file)
    #     self.cited_docs = self._get_cited_docs()
    #     self.evidence_docs = self._get_evidence_docs()

    def __repr__(self):
        msg = f"{self.corpus.__repr__()} {len(self.examples)} claims."
        return msg

    def __getitem__(self, i):
        return self.examples[i]

    def _read_corpus(self, corpus_file):
        corpus = load_jsonl(corpus_file)
        documents = []
        for entry in corpus:
            doc = Document(entry["doc_id"], entry["title"], entry["abstract"])
            documents.append(doc)

        return Corpus(documents)

    def _read_examples(self, data_file):
        examples = load_jsonl(data_file)
        res = []
        for this_example in examples:
            entry = copy.deepcopy(this_example)
            entry["release"] = self
            entry["cited_docs"] = [self.corpus.get_document(doc)
                                   for doc in entry["cited_doc_ids"]]
            assert len(entry["cited_docs"]) == len(entry["cited_doc_ids"])
            del entry["cited_doc_ids"]
            res.append(Example(**entry))

        res = sorted(res, key=lambda x: x.id)
        return res

    def _get_cited_docs(self):
        all_cited_docs = []
        for example in self.examples:
            for cited_doc in example.cited_docs:
                if cited_doc not in all_cited_docs:
                    all_cited_docs.append(cited_doc)

        return sorted(all_cited_docs)

    def _get_evidence_docs(self):
        all_evidence = set()
        for example in self.examples:
            for key in example.evidence:
                all_evidence.add(key)

        evidence_docs = [self.corpus.get_document(id) for id in all_evidence]
        return evidence_docs

    def get_example(self, example_id):
        keep = [x for x in self.examples if x.id == example_id]
        assert len(keep) == 1
        return keep[0]


@dataclass(repr=False)
class Example:
    id: int
    claim: str
    evidence: Dict
    cited_docs: List[Document]
    release: Release

    def __post_init__(self):
        self.evidence = self._format_evidence(self.evidence)

    @staticmethod
    def _format_evidence(evidence_dict):
        res = {}
        for doc_id, rationales in evidence_dict.items():
            doc_id = int(doc_id)
            labels = [x["label"] for x in rationales]
            if len(set(labels)) > 1:
                msg = ("In this SciFact release, each claim / abstract pair "
                       "should only have one label.")
                raise Exception(msg)
            label = make_label(labels[0])
            rationale_sents = [x["sentences"] for x in rationales]
            this_abstract = EvidenceAbstract(doc_id, label, rationale_sents)
            res[doc_id] = this_abstract

        return res

    def __repr__(self):
        msg = f"Example {self.id}: {self.claim}"
        return msg

    def pretty_print(self, evidence_doc_id=None, file=None):
        msg = self.__repr__()
        print(msg, file=file)
        # Print the evidence
        print("\nEvidence sets:", file=file)
        for doc_id, evidence in self.evidence.items():
            # If asked for a specific evidence doc, only show that one.
            if evidence_doc_id is not None and doc_id != evidence_doc_id:
                continue
            print("\n" + 20 * "#" + "\n", file=file)
            ev_doc = self.release.corpus.get_document(doc_id)
            print(f"{doc_id}: {evidence.label.name}", file=file)
            for i, sents in enumerate(evidence.rationales):
                print(f"Set {i}:", file=file)
                kept = [sent for i, sent in enumerate(ev_doc.sentences) if i in sents]
                for entry in kept:
                    print(f"\t- {entry}", file=file)


@dataclass
class EvidenceAbstract:
    id: int
    label: Label
    rationales: List[List[int]]


class PredictedDataset:
    """
    Class to handle predictions.
    """
    def __init__(self, release, rationale_file, entailment_file):
        """
        A prediction is linked to a data release.
        """
        self.release = release
        self.predictions = self._read_predictions(rationale_file, entailment_file)

    def __getitem__(self, i):
        return self.predictions[i]

    def __repr__(self):
        msg = f"Predictions for {len(self.predictions)} claims."
        return msg

    def _read_predictions(self, rationale_file, entailment_file):
        predictions = []

        rationales = load_jsonl(rationale_file)
        entailments = load_jsonl(entailment_file)
        for rationale, entailment in zip(rationales, entailments):
            prediction = self._parse_prediction(rationale, entailment)
            predictions.append(prediction)

        return predictions

    def _parse_prediction(self, rationale, entailment):
        assert rationale["claim_id"] == entailment["claim_id"]
        claim_id = rationale["claim_id"]
        evidences = rationale["evidence"]
        labels = entailment["labels"]

        # Make sure all the keys in labels are in evidences.
        label_keys = set(labels.keys())
        evidence_keys = set(evidences.keys())
        assert label_keys.issubset(evidence_keys)

        preds = {}

        # Deal with the NEI case separately from evidence case.
        for key in evidences:
            if key not in labels:
                label = {"label": "NOT_ENOUGH_INFO", "confidence": 1}
            else:
                label = labels[key]
            evidence = evidences[key]
            pred = Prediction(int(key), make_label(label["label"]), label["confidence"], evidence)
            preds[int(key)] = pred

        return ClaimPredictions(claim_id, preds)


def make_label(label_str):
    lookup = {"SUPPORT": Label.SUPPORTS,
              "NOT_ENOUGH_INFO": Label.NEI,
              "CONTRADICT": Label.REFUTES}
    assert label_str in lookup
    return lookup[label_str]


@dataclass
class Prediction:
    abstract_id: int
    label: Label
    confidence: float
    rationale: List


@dataclass
class ClaimPredictions:
    claim_id: int
    predictions: Dict[int, Prediction]
