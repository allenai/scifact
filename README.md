# SciFact README

This repository contains data and code for the paper [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974).

## Dataset Schema

The data are stored in the `data` subdirectory of this repository. The claims are split into `claims-train.jsonl`, `claims-dev.jsonl`, and `claims-test.jsonl`.

`dataset.jsonl`
```
{
    "id": number,
    "claim": string,
    "label": enum("SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO"),
    "evidence": {
        "doc_id": {
            "sentences": number[]
        }
    },
    "cited_doc_ids": number[]
}
```

`corpus.jsonl`
```
{
    "doc_id": number,
    "title": string,
    "abstract": string[],
    "structured": boolean
}
```

## Model Download

- Project [website](https://github.com/allenai/scifact-demo).
