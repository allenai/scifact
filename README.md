# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974).

- Project [website](https://github.com/allenai/scifact-demo).

## Data

The data are located in the `data` subdirectory of this repository. The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The labels are removed for the test set. The corpus of evidence documents is `corpus.jsonl`, one evidence document per lin. The schemas for the claims and corpus are below.


`claims_(train|dev|test).jsonl` python
```
{
    "id": number,                                                        # An integer claim ID.
    "claim": string,                                                     # The text of the claim.
    "label": enum("SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO"),         # The claim's label.
    "evidence": {                                                        # The evidence for the claim.
        "doc_id": {                                                      # GORC ID of evidence document.
            "sentences": number[]                                        # Rationale sentences.
        }
    },
    "cited_doc_ids": number[]                                            # The claim's "cited documents".
}
```

NOTE: In the task definition as presented in the [paper](https://arxiv.org/abs/2004.14974), a single claim may have both supporting and contradictory evidence documents. However, these claims are not included in our initial dataset release. Therefore, the claim is given a single label; all evidence documents listed have the same label with respect to the claim.

`corpus.jsonl` python
```
{
    "doc_id": number,                                                    # The document's S2ORC ID.
    "title": string,                                                     # The title.
    "abstract": string[],                                                # The abstract, written as a list of sentences.
    "structured": boolean                                                # Indicator for whether this is a structured abstract.
}
```

## Model Download
