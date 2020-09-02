# Leaderboard submissions

## Submission format

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


## Result format

After I run your submission, I'll send you back a `tar.gz` file named like `results-[submission-date].tar.gz`. When upacked, it will be organized like:

- `[submission-date]`
  - `predictions`
    - All prediction files submitted on the given date (as `.jsonl`).
  - `results`
    - Test set performance. For each prediction file, there will be a matching `.tsv` file here.
  - `log`
    - Output from `stderr`. If not empty, it probably contains warnings about predictions with more than 3 rationale sentences.

For example:

- `[2020-09-02]`
  - `predictions`
    - `model-1.jsonl`
    - `model-2.jsonl`
  - `results`
    - `model-1.tsv`
    - `model-2.tsv`
  - `log`
    - `model-1.log`
    - `model-2.log`
