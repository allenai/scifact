import argparse
import torch
import jsonlines
import os

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dev', type=str, required=True, help='Path to processed test file')
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-3)
args = parser.parse_args()


class FeverRationaleSelectionDataset(Dataset):
    def __init__(self, file):
        self.samples = []
        for data in jsonlines.open(file):
            evidence_indices = {s for es in data['evidence_sets'] for s in es}
            self.samples.extend([{
                'claim': data['claim'],
                'sentence': s,
                'evidence': i in evidence_indices
            } for i, s in enumerate(data['sentences'])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

trainset = FeverRationaleSelectionDataset(args.train)
devset = FeverRationaleSelectionDataset(args.dev)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
optimizer = torch.optim.Adam([
    {'params': model.roberta.parameters(), 'lr': args.lr_base},  # if using non-roberta model, change the base param path.
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)


def encode(claims: List[str], sentences: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(sentences, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(sentences, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset, full=False):
    model.eval()
    targets = []
    outputs = []
    if not full:
        dataset = Subset(dataset, range(0, 200 * 16))
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=16)):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return f1_score(targets, outputs, zero_division=0), \
           precision_score(targets, outputs, zero_division=0), \
           recall_score(targets, outputs, zero_division=0)


for e in range(args.epochs):
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['sentence'])
        loss, logits = model(**encoded_dict, labels=batch['evidence'].long().to(device))
        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    train_score = evaluate(model, trainset)
    print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
    dev_score = evaluate(model, devset)
    print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
    # Save
    save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
