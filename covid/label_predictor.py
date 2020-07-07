import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List


class LabelPredictor:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.labels = ['REFUTE', 'NOT_ENOUGH_INFO', 'SUPPORT']

    def __call__(self, claim: str, retrievals: List[dict]):
        results = []
        with torch.no_grad():
            for retrieval in retrievals:
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
                if label_index != 1:
                    result = retrieval.copy()
                    result['label'] = self.labels[label_index]
                    result['label_confidence'] = label_scores[0, label_index].item()
                    results.append(result)
        return results
