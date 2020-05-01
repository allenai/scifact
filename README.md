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
        "doc_id": [                                                      # The rationales for a single document, keyed by S2ORC ID.
            {
            "label": string,                                             # Label for the rationale.
            "sentences": number[]                                        # Rationale sentences.
            }
        ]
    },
    "cited_doc_ids": number[]                                            # The claim's "cited documents".
}
```

NOTE: In our dataset, the labels for all rationales within a single abstract will be the same.

`corpus.jsonl` python
```
{
    "doc_id": number,                                                    # The document's S2ORC ID.
    "title": string,                                                     # The title.
    "abstract": string[],                                                # The abstract, written as a list of sentences.
    "structured": boolean                                                # Indicator for whether this is a structured abstract.
}
```
