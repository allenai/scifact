import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List


class RationaleSelector:
    def __init__(self, model: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)

    def __call__(self, claim: str, documents: List[dict], k=3):
        results = []
        with torch.no_grad():
            for document in documents:
                encoded_dict = self.tokenizer.batch_encode_plus(
                    zip(document['abstract'], [claim] * len(document['abstract'])),
                    pad_to_max_length=True,
                    return_tensors='pt'
                )
                encoded_dict = {key: tensor.to(self.device) for key, tensor in encoded_dict.items()}
                evidence_logits = self.model(**encoded_dict)[0]
                evidence_scores = torch.sigmoid(evidence_logits[:, 1]).cpu().numpy()
                evidence_indices = list(sorted(evidence_scores.argsort()[-k:][::-1].tolist()))
                document = document.copy()
                document['evidence'] = evidence_indices
                document['evidence_confidence'] = evidence_scores[evidence_indices].tolist()
                results.append(document)
        return results
