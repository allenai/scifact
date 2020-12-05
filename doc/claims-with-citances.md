The file `claims_with_citances.jsonl` contains data to allow researchers to train "claim generation" models: given a citation context, generate self-contained "atomic" claims based on the context.

Each line in the file contains information on the claims written based on a single citaion sentence. The fiels are:

- `s2orc_id`: The S2ORC document ID of the document containing the citance.
- `title`: The title of the paper.
- `abstract`: The abstract of the paper.
- `citation_paragraph`: The paragraph containing the citation sentence.
- `citance`: The citation sentence on which the claim is based.
- `claims`: A list of claims, written based on the citance. Each claim has the following fields:
  - `claim_text`: The text of the claim.
  - `claim_subject`: The subject of the claim. May be blank.
  - `is_negation`: Does this claim negate the meaning of the citance?
