# Leaderboard submission format

Each line should contain a prediction for a single test set document. The schema is similar to that of the gold data described in [data.md](data.md), but simplified as follows: for each evidence document, the model is asked to predict a single label and a list of rationale sentences, rather than predicting rationales explicitly. More details can be found in the [paper](https://arxiv.org/abs/2004.14974).


```python
{
    "id": number,                   # An integer claim ID.
    "evidence": {                   # The evidence for the claim.
        [doc_id]: {                 # The sentences and label for a single document, keyed by S2ORC ID.
            "label": enum("SUPPORT" | "CONTRADICT"),
            "sentences": number[]
        }
    },
}
```

Note that `evidence` may be empty.

Here's an example claim with two predicted evidence documents:
```python
{
    'id': 84,
    'evidence': {
        '22406695': {'sentences': [1], 'label': 'SUPPORT'},
        '7521113': {'sentences': [4], 'label': 'SUPPORT'}
    }
}
```

Run `./script/pipeline.sh oracle oracle-rationale test` and examine `predictions/merged_predictions.jsonl` to see an example of predictions on the full test set.
