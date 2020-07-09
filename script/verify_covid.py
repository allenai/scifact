import torch
import argparse
from subprocess import call
from copy import deepcopy

from covid import AbstractRetriever, RationaleSelector, LabelPredictor


def get_args():
    parser = argparse.ArgumentParser(
        description="Verify a claim against the CORD-19 corpus.")
    parser.add_argument("claim", type=str, help="The claim to be verified")
    parser.add_argument("report_file", type=str, default="report",
                        help="The file where the report will be written (no file extension).")
    parser.add_argument("--n_documents", type=int, default=20,
                        help="The number of documents to retrieve.")
    parser.add_argument("--rationale_selection_method", type=str,
                        choices=["topk", "threshold"], default="topk",
                        help="Select top k rationales, or keep sentences above score threshold.")
    parser.add_argument("--rationale_threshold", type=float, default=0.5,
                        help="Classification threshold for selecting rationale sentences.")
    parser.add_argument("--keep_nei", action="store_true",
                        help="Keep examples labeled `NOT_ENOUGH_INFO`.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def inference(claim: str, n_documents: int, rationale_selection_method: str,
              rationale_threshold: float, keep_nei: bool, verbose: bool):
    # Initialize pipeline components
    if verbose:
        print("Initializing model.")
    rationale_selection_model = 'model/rationale_roberta_large_scifact'
    label_prediction_model = 'model/label_roberta_large_fever_scifact'
    abstract_retriever = AbstractRetriever()
    rationale_selector = RationaleSelector(rationale_selection_model,
                                           rationale_selection_method,
                                           rationale_threshold)
    label_predictor = LabelPredictor(label_prediction_model, keep_nei)

    # Run model.
    if verbose:
        print("Retrieving abstracts.")
    results = abstract_retriever(claim, k=n_documents)

    if verbose:
        print("Selecting rationales.")
    results = rationale_selector(claim, results)

    if verbose:
        print("Making label predictions")
    results = label_predictor(claim, results)

    results.sort(key=lambda r: r['label_confidence'], reverse=True)
    return results


def write_result(result, f):
    msg = f"### {result['title']}\n"
    print(msg, file=f)
    ev_scores = [f"{x:0.2f}" for x in result["evidence_confidence"]]
    ev_scores = ", ".join(ev_scores)
    msg = f"**Decision**: {result['label']} (score={result['label_confidence']:0.2f}, evidence scores={ev_scores})\n"
    print(msg, file=f)

    msg = "#### Evidence\n"
    print(msg, file=f)
    for i, line in enumerate(result["abstract"]):
        if i in result["evidence"]:
            msg = f"- {line}"
            print(msg, file=f)
    print(file=f)

    msg = "#### Metadata\n"
    print(msg, file=f)
    msg = (f"- Journal: {result['journal']}\n"
           f"- Authors: {', '.join(result['authors'])}\n"
           f"- URL: {result['url']}\n")
    print(msg, file=f)


def export(args, results):
    claim = args.claim
    report_file = args.report_file
    f = open(f"{report_file}.md", "w")
    msg = f"## Claim\n {claim}"
    print(msg, file=f)
    print(file=f)

    msg = "## Evidence\n"
    print(msg, file=f)
    for result in results:
        write_result(result, f)

    msg = "## Config\n"
    print(msg, file=f)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        if isinstance(v, float):
            v = f"{v:0.2f}"
        msg = f"- {k}: {v}"
        print(msg, file=f)

    f.close()
    pdf_file = f"{report_file}.pdf"
    cmd = ["pandoc", f"{report_file}.md", "-o", pdf_file, "-t", "html"]
    call(cmd)


################################################################################

if __name__ == "__main__":
    args = get_args()
    inference_args = deepcopy(vars(args))
    del inference_args["report_file"]
    results = inference(**inference_args)
    export(args, results)
