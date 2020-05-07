"""
Code to represent a dataset release.
"""

from enum import Enum
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

####################

# Utility functions and enums.


def load_jsonl(fname):
    return [json.loads(line) for line in open(fname)]


class Label(Enum):
    SUPPORTS = 1
    NEI = 0
    REFUTES = -1


def make_label(label_str):
    lookup = {"SUPPORT": Label.SUPPORTS,
              "NOT_ENOUGH_INFO": Label.NEI,
              "CONTRADICT": Label.REFUTES}
    assert label_str in lookup
    return lookup[label_str]


####################

# Representations for the corpus and abstracts.

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


####################

# Gold dataset.

class GoldDataset:
    """
    Class to represent a gold dataset, include corpus and claims.
    """
    def __init__(self, corpus_file, data_file):
        self.corpus = self._read_corpus(corpus_file)
        self.claims = self._read_claims(data_file)

    def __repr__(self):
        msg = f"{self.corpus.__repr__()} {len(self.examples)} claims."
        return msg

    def __getitem__(self, i):
        return self.claims[i]

    def _read_corpus(self, corpus_file):
        "Read corpus from file."
        corpus = load_jsonl(corpus_file)
        documents = []
        for entry in corpus:
            doc = Document(entry["doc_id"], entry["title"], entry["abstract"])
            documents.append(doc)

        return Corpus(documents)

    def _read_claims(self, data_file):
        "Read claims from file."
        examples = load_jsonl(data_file)
        res = []
        for this_example in examples:
            entry = copy.deepcopy(this_example)
            entry["release"] = self
            entry["cited_docs"] = [self.corpus.get_document(doc)
                                   for doc in entry["cited_doc_ids"]]
            assert len(entry["cited_docs"]) == len(entry["cited_doc_ids"])
            del entry["cited_doc_ids"]
            res.append(Claim(**entry))

        res = sorted(res, key=lambda x: x.id)
        return res

    def get_claim(self, example_id):
        "Get a single claim by ID."
        keep = [x for x in self.claims if x.id == example_id]
        assert len(keep) == 1
        return keep[0]


@dataclass
class EvidenceAbstract:
    "A single evidence abstract."
    id: int
    label: Label
    rationales: List[List[int]]


@dataclass(repr=False)
class Claim:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """
    id: int
    claim: str
    evidence: Dict[int, EvidenceAbstract]
    cited_docs: List[Document]
    release: GoldDataset

    def __post_init__(self):
        self.evidence = self._format_evidence(self.evidence)

    @staticmethod
    def _format_evidence(evidence_dict):
        # This function is needed because the data schema is designed so that
        # each rationale can have its own support label. But, in the dataset,
        # all rationales for a given claim / abstract pair all have the same
        # label. So, we store the label at the "abstract level" rather than the
        # "rationale level".
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
        "Pretty-print the claim, together with all evidence."
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


####################

# Predicted dataset.

class PredictedDataset:
    """
    Class to handle predictions, with a pointer back to the gold data.
    """
    def __init__(self, gold, rationale_file, label_file):
        """
        Takes a GoldDataset, as well as files with rationale and label
        predictions.
        """
        self.gold = gold
        self.predictions = self._read_predictions(rationale_file, label_file)

    def __getitem__(self, i):
        return self.predictions[i]

    def __repr__(self):
        msg = f"Predictions for {len(self.predictions)} claims."
        return msg

    def _read_predictions(self, rationale_file, label_file):
        predictions = []

        rationales = load_jsonl(rationale_file)
        labels = load_jsonl(label_file)
        for rationale, label in zip(rationales, labels):
            prediction = self._parse_prediction(rationale, label)
            predictions.append(prediction)

        return predictions

    def _parse_prediction(self, rationale, label):
        assert rationale["claim_id"] == label["claim_id"]
        claim_id = rationale["claim_id"]
        evidences = rationale["evidence"]
        labels = label["labels"]

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
            pred = PredictedAbstract(int(key), make_label(label["label"]),
                                     label["confidence"], evidence)
            preds[int(key)] = pred

        return ClaimPredictions(claim_id, preds)


@dataclass
class PredictedAbstract:
    # For predictions, we have a single list of rationale sentences instead of a
    # list of separate rationales (see paper for details).
    abstract_id: int
    label: Label
    confidence: float
    rationale: List


@dataclass
class ClaimPredictions:
    claim_id: int
    predictions: Dict[int, PredictedAbstract]
