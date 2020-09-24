import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
from tqdm import tqdm


class LabelPredictor:
    def __init__(self, model: str, keep_nei: bool, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.labels = ['REFUTE', 'NOT_ENOUGH_INFO', 'SUPPORT']
        self.keep_nei = keep_nei

    def __call__(self, claim: str, retrievals: List[dict]):
        print("Predicting labels.")
        results = []
        with torch.no_grad():
            for retrieval in tqdm(retrievals):
                if not retrieval['evidence']:
                    continue
                sentences = [retrieval['abstract'][i] for i in retrieval['evidence']]
                encoded_dict = self.tokenizer.batch_encode_plus(
                    [(' '.join(sentences), claim)],
                    return_tensors='pt'
                )
                encoded_dict = {key: tensor.to(self.device) for key, tensor in encoded_dict.items()}
                label_scores = self.model(**encoded_dict)[0].softmax(dim=1)
                label_index = label_scores.argmax(dim=1).item()

                # If `keep_nei` is true, keep all evidence.
                keep = self.keep_nei or (label_index != 1)
                if keep:
                    result = retrieval.copy()
                    result['label'] = self.labels[label_index]
                    result['label_confidence'] = label_scores[0, label_index].item()
                    results.append(result)
        return results
