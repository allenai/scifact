"""
Evaluating abstract-level and sentence-level performance as desentenced in the
paper.
"""

from lib.data import Label
from collections import Counter
import pandas as pd


def compute_f1(counts):
    precision = counts["correct"] / counts["retrieved"]
    recall = counts["correct"] / counts["relevant"]
    f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


####################

# Abstract-level evaluation

def contains_evidence(predicted, gold):
    # If any of gold are contained in predicted, we're good.
    for gold_rat in gold:
        if gold_rat.issubset(predicted):
            return True
    # If we get to the end, didn't find one.
    return False


def is_correct(doc_id, doc_pred, gold):
    # If it's not an evidence document, we lose.
    if doc_id not in gold.evidence:
        return False

    # If the label's wrong, we lose.
    gold_label = gold.evidence[doc_id].label
    if doc_pred.label != gold_label:
        return False

    gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
    # Otherwise, we win if it's got an evidence set.
    return contains_evidence(set(doc_pred.rationale), gold_rationales)


def update_counts_abstract(pred, gold, counts_abstract):
    counts_abstract["relevant"] += len(gold.evidence)
    for doc_id, doc_pred in pred.predictions.items():
        # If it's NEI, doesn't count one way or the other.
        if doc_pred.label == Label.NEI:
            continue
        counts_abstract["retrieved"] += 1
        good = is_correct(doc_id, doc_pred, gold)
        if good:
            counts_abstract["correct"] += 1

    return counts_abstract


####################

# Sentence-level evaluation

def count_rationale_sents(predicted, gold):
    n_correct = 0

    for ix in predicted:
        gold_sets = [entry for entry in gold if ix in entry]
        assert len(gold_sets) < 2  # Can't be in two rationales.
        # If it's not in a gold set, no dice.
        if len(gold_sets) == 0:
            continue
        # If it's in a gold set, make sure the rest got retrieved.
        gold_set = gold_sets[0]
        if gold_set.issubset(predicted):
            n_correct += 1

    return n_correct


def count_correct(doc_id, doc_pred, gold):
    # If not an evidence doc, no good.
    if doc_id not in gold.evidence:
        return 0
    # If label is wrong, no good.
    gold_label = gold.evidence[doc_id].label
    if doc_pred.label != gold_label:
        return 0

    # Count the number of rationale sentences we get credit for.
    gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
    n_correct = count_rationale_sents(set(doc_pred.rationale), gold_rationales)
    return n_correct


def update_counts_sentence(pred, gold, counts_sentence):
    # Update the gold evidence sentences.
    for gold_doc in gold.evidence.values():
        counts_sentence["relevant"] += sum([len(x) for x in gold_doc.rationales])

    for doc_id, doc_pred in pred.predictions.items():
        # If it's NEI, skip it.
        if doc_pred.label == Label.NEI:
            continue

        counts_sentence["retrieved"] += len(doc_pred.rationale)
        n_correct = count_correct(doc_id, doc_pred, gold)
        counts_sentence["correct"] += n_correct

    return counts_sentence


################################################################################

def compute_metrics(preds):
    """
    Compute pipeline metrics based on dataset of predictions.
    """
    counts_abstract = Counter()
    counts_sentence = Counter()

    for pred in preds:
        gold = preds.gold.get_claim(pred.claim_id)
        counts_abstract = update_counts_abstract(pred, gold, counts_abstract)
        counts_sentence = update_counts_sentence(pred, gold, counts_sentence)

    f1_abstract = compute_f1(counts_abstract)
    f1_sentence = compute_f1(counts_sentence)

    return pd.DataFrame({"abstract": f1_abstract, "sentence": f1_sentence})
