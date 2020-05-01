These schemas for the claim and corpus data are as follows:

`claims_(train|dev|test).jsonl`
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

`corpus.jsonl`
```
{
    "doc_id": number,                                                    # The document's S2ORC ID.
    "title": string,                                                     # The title.
    "abstract": string[],                                                # The abstract, written as a list of sentences.
    "structured": boolean                                                # Indicator for whether this is a structured abstract.
}
```
