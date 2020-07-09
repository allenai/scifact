import argparse
from subprocess import call
import os

from verisci.covid import AbstractRetriever, RationaleSelector, LabelPredictor


def get_args():
    parser = argparse.ArgumentParser(
        description="Verify a claim against the CORD-19 corpus.")
    parser.add_argument("claim", type=str, help="The claim to be verified")
    parser.add_argument("report_file", type=str, default="report",
                        help="The file where the report will be written (no file extension).")
    parser.add_argument("--n_documents", type=int, default=20,
                        help="The number of documents to retrieve from Covidex.")
    parser.add_argument("--rationale_selection_method", type=str,
                        choices=["topk", "threshold"], default="topk",
                        help="Select top k rationales, or keep sentences with sigmoid scores above `rationale_threshold`.")
    parser.add_argument("--output_format", type=str,
                        choices=["pdf", "markdown"], default="pdf",
                        help="Output format. PDF requires Pandoc.")
    parser.add_argument("--rationale_threshold", type=float, default=0.5,
                        help="Classification threshold for selecting rationale sentences.")
    parser.add_argument("--keep_nei", action="store_true",
                        help="Keep examples labeled `NOT_ENOUGH_INFO`, for which evidence was identified.")
    parser.add_argument("--full_abstract", action="store_true",
                        help="Show full abstracts, not just evidence sentences.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose model output.")
    return parser.parse_args()


def inference(args):
    # Initialize pipeline components
    if args.verbose:
        print("Initializing model.")
    rationale_selection_model = 'model/rationale_roberta_large_scifact'
    label_prediction_model = 'model/label_roberta_large_fever_scifact'
    abstract_retriever = AbstractRetriever()
    rationale_selector = RationaleSelector(rationale_selection_model,
                                           args.rationale_selection_method,
                                           args.rationale_threshold)
    label_predictor = LabelPredictor(label_prediction_model, args.keep_nei)

    # Run model.
    if args.verbose:
        print("Retrieving abstracts.")
    results = abstract_retriever(args.claim, k=args.n_documents)

    if args.verbose:
        print("Selecting rationales.")
    results = rationale_selector(args.claim, results)

    if args.verbose:
        print("Making label predictions")
    results = label_predictor(args.claim, results)

    results.sort(key=lambda r: r['label_confidence'], reverse=True)
    return results


def write_result(result, f, full_abstract):
    msg = f"#### [{result['title']}]({result['url']})"
    print(msg, file=f)
    ev_scores = [f"{x:0.2f}" for x in result["evidence_confidence"]]
    ev_scores = ", ".join(ev_scores)
    msg = f"**Decision**: {result['label']} (score={result['label_confidence']:0.2f}, evidence scores={ev_scores})\n"
    print(msg, file=f)

    for i, line in enumerate(result["abstract"]):
        # If we're showing the full abstract, show evidence in green.
        if full_abstract:
            msg = (f"- <span style='color:green'>{line}</span>"
                   if i in result["evidence"]
                   else f"- {line}")
            print(msg, file=f)
        else:
            if i in result["evidence"]:
                msg = f"- {line}"
                print(msg, file=f)

    print(file=f)
    print(40 * "-", file=f)
    print(file=f)


def export(args, results):
    claim = args.claim
    report_file = args.report_file
    f = open(f"{report_file}.md", "w")
    msg = f"### Claim\n {claim}"
    print(msg, file=f)
    print(file=f)

    msg = "### Evidence\n"
    print(msg, file=f)
    for result in results:
        write_result(result, f, args.full_abstract)

    msg = "## Config\n"
    print(msg, file=f)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        if isinstance(v, float):
            v = f"{v:0.2f}"
        msg = f"- {k}: {v}"
        print(msg, file=f)

    f.close()
    if args.output_format == "pdf":
        pdf_file = f"{report_file}.pdf"
        cmd = ["pandoc", f"{report_file}.md", "-o", pdf_file, "-t", "html"]
        call(cmd)
        os.remove(f"{report_file}.md")


def main():
    args = get_args()
    results = inference(args)
    export(args, results)


if __name__ == "__main__":
    main()
