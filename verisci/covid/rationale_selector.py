import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from tqdm import tqdm


class RationaleSelector:
    def __init__(self, model: str, selection_method: str, threshold: float,
                 device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.selection_method = selection_method
        self.threshold = threshold

    def __call__(self, claim: str, documents: List[dict], k=3):
        print("Selecting rationales.")
        results = []
        with torch.no_grad():
            for document in tqdm(documents):
                encoded_dict = self.tokenizer.batch_encode_plus(
                    zip(document['abstract'], [claim] * len(document['abstract'])),
                    pad_to_max_length=True,
                    return_tensors='pt'
                )
                encoded_dict = {key: tensor.to(self.device) for key, tensor in encoded_dict.items()}
                evidence_logits = self.model(**encoded_dict)[0]
                evidence_scores = torch.sigmoid(evidence_logits[:, 1]).cpu().numpy()

                if self.selection_method == "threshold":
                    keep = evidence_scores > self.threshold
                    evidence_indices = sorted(keep.nonzero()[0].tolist())
                else:
                    evidence_indices = list(sorted(evidence_scores.argsort()[-k:][::-1].tolist()))

                document = document.copy()
                document['evidence'] = evidence_indices
                document['evidence_confidence'] = evidence_scores[evidence_indices].tolist()
                results.append(document)
        return results
