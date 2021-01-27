# SciFact leaderboard submission and evaluation guidelines

- [Submission format](#submission-format)
- [Detailed scoring example](#detailed-scoring-example)

## Submission format

Each line in a leaderboard submission should contain a prediction for a single test set document. The schema is similar to that of the gold data described in [data.md](data.md), but simplified as follows: for each evidence document, the model is asked to predict a single label and a list of rationale sentences, rather than predicting rationales explicitly. More details can be found in the [paper](https://arxiv.org/abs/2004.14974).


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

## Detailed scoring example

In this example, we walk through the process of evaluating the predictions for a single claim.

### Gold

The claim has evidence in two abstracts. Abstract `11` has two separate evidence sets that serve to verify the claim while abstract `15` has one.

```javascript
{
  "claim": "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
  "evidence": {
      "11": [                           // 2 evidence sets in document 11 support the claim.
          { "sentences": [ 0, 1 ],      // Sentences 0 and 1, taken together, support the claim.
             "label": "SUPPORT" },
          { "sentences": [ 11 ],        // Sentence 11, on its own, supports the claim.
             "label": "SUPPORT" }
      ],
      "15": [                           // A single evidence set in document 15 supports the claim.
          { "sentences": [ 4 ],
             "label": "SUPPORT" }
      ]
  }
}
```

### Predicted

The model predicts two abstracts are relevant to the claim. For each abstract, it provides a label and a list of sentences as evidence sentences.

```javascript
{
  "claim": "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
  "evidence": {
      "11": {
          "sentences": [ 1, 11, 13 ],     // Predicted rationale sentences.
          "label": "SUPPORT"              // Predicted label.
      },
      "16": {
        "sentences": [18, 20 ],
        "label": "REFUTES"
      }
  }
}
```

### Abstract-level scoring

An abstract is _correctly predicted_ if (1) it is a relevant abstract, (2) its predicted label matches gold label, and (3) at least one gold evidence set is contained among the predicted evidence sentences.

Note that our abstract evaluation code only counts the **first three predicted evidence sentences** for a given abstract; additional sentences are ignored. This is similar to the FEVER score, and is required because abstract-level evaluation does not penalize the model for over-predicting rationale sentences.

Precision and recall are defined as follows:

- **Precision**: `(# correctly predicted abstracts) / (# predicted abstracts)`
- **Recall**: `(# correctly predicted abstracts) / (# gold abstracts)`

Let's count the number of predicted and gold abstracts:

- **# predicted abstracts**: 2 (abstract 11, abstract 16)
- **# gold abstracts**: 2 (abstract 11, abstract 15)

Now, count how many of the predicted abstracts are correct:
​
- **11** is _correct_, since:
  - The predicted label matches the gold label (SUPPORTS).
  - Gold evidence set `[ 11 ]` is contained in the predicted rationale sentences `[1, 11, 13]`.
- **16** is _incorrect_, since:
  - Abstract 16 is not in the gold set of relevant abstracts.

Finally, calculate precision, recall, and F1:

- **Precision** = `1 / 2`
- **Recall** = `1 / 2`
- **F1**  = `1 / 2`

### Sentence-level scoring

An evidence sentence is _correctly predicted_ if (1) its from a relevant abstract, (2) the label of its abstract matches gold label, and (3) it is part of some gold evidence set, and (3) all other sentences in that same gold evidence set are _also_ among the predicted evidence sentences.

For sentence-level scoring, **all predicted evidence sentences are counted** (unlike abstract evaluation).

Precision and recall are defined as follows:

- **Precision**: `(# correctly predicted evidence sentences) / (# predicted evidence sentences)`
- **Recall**: `(# correctly predicted evidence sentences) / (# gold evidence sentences)`

Let's count the predicted and gold evidence sentences:

- **# predicted evidence sentences**: 3 (abstract 11) + 2 (abstract 16) = 5 total
- **# gold evidence sentences**: 3 (abstract 11) + 2 (abstract 15) = 4 total

Now, count the number of correctly predicted evidence sentences:

- Abstract 11
  - **Sentence 1**: _Incorrect_. It is part of the gold evidence set `[0, 1]`, but failed to also predict Sentence `0`. Since the whole evidence set `[0, 1]` was not predicted, the model does not get credit for Sentence `1`.
  - **Sentence 11**: _Correct_. It is part of the gold evidence set `[11]`.  Model is not not missing any other sentences in that gold evidence set.
  - **Sentence 13**: _Incorrect_. It is not part of any gold evidence.
- Abstract 16
  - **Sentences 18, 20**: _Incorrect_. Abstract 16 is not in gold set of relevant abstracts.

Calculate precision, recall, and F1:

- **Precision** = `1 / 5`
- **Recall** = `1 / 4`
- **F1** = `2 / 9`
