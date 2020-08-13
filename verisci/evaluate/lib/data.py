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


def make_label(label_str, allow_NEI=True):
    lookup = {"SUPPORT": Label.SUPPORTS,
              "NOT_ENOUGH_INFO":  Label.NEI,
              "CONTRADICT": Label.REFUTES}

    res = lookup[label_str]
    if (not allow_NEI) and (res is Label.NEI):
        raise ValueError("An NEI was given.")

    return res


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
        msg = f"{self.corpus.__repr__()} {len(self.claims)} claims."
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
    def __init__(self, gold, prediction_file):
        """
        Takes a GoldDataset, as well as files with rationale and label
        predictions.
        """
        self.gold = gold
        self.predictions = self._read_predictions(prediction_file)

    def __getitem__(self, i):
        return self.predictions[i]

    def __repr__(self):
        msg = f"Predictions for {len(self.predictions)} claims."
        return msg

    def _read_predictions(self, prediction_file):
        res = []

        predictions = load_jsonl(prediction_file)
        for pred in predictions:
            prediction = self._parse_prediction(pred)
            res.append(prediction)

        return res

    def _parse_prediction(self, pred_dict):
        claim_id = pred_dict["id"]
        predicted_evidence = pred_dict["evidence"]

        res = {}

        # Predictions should never be NEI; there should only be predictions for
        # the abstracts that contain evidence.
        for key, this_prediction in predicted_evidence.items():
            label = this_prediction["label"]
            evidence = this_prediction["sentences"]
            pred = PredictedAbstract(int(key),
                                     make_label(label, allow_NEI=False),
                                     evidence)
            res[int(key)] = pred

        return ClaimPredictions(claim_id, res)


@dataclass
class PredictedAbstract:
    # For predictions, we have a single list of rationale sentences instead of a
    # list of separate rationales (see paper for details).
    abstract_id: int
    label: Label
    rationale: List


@dataclass
class ClaimPredictions:
    claim_id: int
    predictions: Dict[int, PredictedAbstract]
