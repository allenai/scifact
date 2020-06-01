"""
Evaluating abstract-level and sentence-level performance as described in the
paper.
"""

import warnings

from lib.data import Label
from collections import Counter
import pandas as pd


# Cap on how many abstract sentences can be returned.
MAX_ABSTRACT_SENTS = 3


def compute_f1(counts, difficulty):
    precision = counts[f"correct_{difficulty}"] / counts["retrieved"]
    recall = counts[f"correct_{difficulty}"] / counts["relevant"]
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
    pred_rationales = doc_pred.rationale[:MAX_ABSTRACT_SENTS]

    # If it's not an evidence document, we lose.
    if doc_id not in gold.evidence:
        return False, False

    # If the label's wrong, we lose.
    gold_label = gold.evidence[doc_id].label
    if doc_pred.label != gold_label:
        return False, False

    gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
    good_rationalized = contains_evidence(set(pred_rationales), gold_rationales)
    good_label_only = True
    return good_label_only, good_rationalized


def update_counts_abstract(pred, gold, counts_abstract):
    counts_abstract["relevant"] += len(gold.evidence)
    for doc_id, doc_pred in pred.predictions.items():
        # If it's NEI, doesn't count one way or the other.
        if doc_pred.label == Label.NEI:
            continue
        counts_abstract["retrieved"] += 1

        good_label_only, good_rationalized = is_correct(doc_id, doc_pred, gold)
        if good_label_only:
            counts_abstract["correct_label_only"] += 1
        if good_rationalized:
            counts_abstract["correct_rationalized"] += 1

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
        return 0, 0

    # Count the number of rationale sentences we get credit for.
    gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
    n_correct = count_rationale_sents(set(doc_pred.rationale), gold_rationales)

    gold_label = gold.evidence[doc_id].label

    n_correct_selection = n_correct
    correct_label = int(doc_pred.label == gold_label)
    n_correct_label = correct_label * n_correct

    return n_correct_selection, n_correct_label


def update_counts_sentence(pred, gold, counts_sentence):
    # Update the gold evidence sentences.
    for gold_doc in gold.evidence.values():
        counts_sentence["relevant"] += sum([len(x) for x in gold_doc.rationales])

    for doc_id, doc_pred in pred.predictions.items():
        # If it's NEI, skip it.
        if doc_pred.label == Label.NEI:
            continue

        counts_sentence["retrieved"] += len(doc_pred.rationale)
        n_correct_selection, n_correct_label = count_correct(doc_id, doc_pred, gold)
        counts_sentence["correct_selection"] += n_correct_selection
        counts_sentence["correct_label"] += n_correct_label

    return counts_sentence


####################

# Make sure rationales aren't too long.

def check_rationale_lengths(preds):
    bad = []
    for pred in preds:
        claim_id = pred.claim_id
        predictions = pred.predictions
        for doc_key, prediction in predictions.items():
            n_rationales = len(prediction.rationale)
            if n_rationales > MAX_ABSTRACT_SENTS:
                to_append = {"claim_id": claim_id, "abstract": doc_key, "n_rationales": n_rationales}
                bad.append(to_append)
    if bad:
        bad = pd.DataFrame(bad)
        msg = (f"\nRationales with more than {MAX_ABSTRACT_SENTS} sentences found.\n"
               f"The first 3 will be used for abstract-level evaluation\n\n"
               f"{bad.__repr__()}")
        warnings.warn(msg)
        print()


################################################################################

def compute_metrics(preds):
    """
    Compute pipeline metrics based on dataset of predictions.
    """
    counts_abstract = Counter()
    counts_sentence = Counter()

    check_rationale_lengths(preds)

    for pred in preds:
        gold = preds.gold.get_claim(pred.claim_id)
        counts_abstract = update_counts_abstract(pred, gold, counts_abstract)
        counts_sentence = update_counts_sentence(pred, gold, counts_sentence)

    return pd.DataFrame(
        {"abstract_label_only": compute_f1(counts_abstract, "label_only"),
         "abstract_rationalied": compute_f1(counts_abstract, "rationalized"),
         "sentence_selection": compute_f1(counts_sentence, "selection"),
         "sentence_label": compute_f1(counts_sentence, "label")})
