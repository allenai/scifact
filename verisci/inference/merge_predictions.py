import argparse
import json


NEI_LABEL = "NOT_ENOUGH_INFO"


def get_args():
    desc = "Merge predictions of rationale selection and label prediction modules."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--rationale-file", type=str, required=True,
                        help="File with rationale predictions.")
    parser.add_argument("--label-file", type=str, required=True,
                        help="File with label predictions.")
    parser.add_argument("--result-file", type=str, required=True,
                        help="File to write merged results to.")
    args = parser.parse_args()
    return args


def merge_one(rationale, label):
    """
    Merge a single rationale / label pair. Throw out NEI predictions.
    """
    evidence = rationale["evidence"]
    labels = label["labels"]
    claim_id = rationale["claim_id"]

    # Check that the documents match.
    if evidence.keys() != labels.keys():
        raise ValueError(f"Evidence docs for rationales and labels don't match for claim {claim_id}.")

    docs = sorted(evidence.keys())

    final_predictions = {}

    for this_doc in docs:
        this_evidence = evidence[this_doc]
        this_label = labels[this_doc]["label"]

        if this_label != NEI_LABEL:
            final_predictions[this_doc] = {"sentences": this_evidence,
                                           "label": this_label}

    res = {"id": claim_id,
           "evidence": final_predictions}
    return res


def merge(rationale_file, label_file, result_file):
    """
    Merge rationales with predicted labels.
    """
    rationales = [json.loads(line) for line in open(rationale_file)]
    labels = [json.loads(line) for line in open(label_file)]

    # Check the ordering
    rationale_ids = [x["claim_id"] for x in rationales]
    label_ids = [x["claim_id"] for x in labels]
    if rationale_ids != label_ids:
        raise ValueError("Claim ID's for label and rationale file don't match.")

    res = [merge_one(rationale, label)
           for rationale, label in zip(rationales, labels)]

    with open(result_file, "w") as f:
        for entry in res:
            print(json.dumps(entry), file=f)


def main():
    args = get_args()
    merge(**vars(args))


if __name__ == "__main__":
    main()
