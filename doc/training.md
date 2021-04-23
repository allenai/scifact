# Training Scripts

All the training scripts are located in the `/verisci/training` folder

- [Common Arguments](#common-arguments)
- [Training on SciFact](#training-on-scifact-dataset)
- [Training on FEVER](#training-on-fever-dataset)
- [Training on FEVER + SciFact](#training-on-fever--scifact-dataset)
- [Training on Snopes](#training-on-snopes-dataset)

## Common Arguments

* `--model`: Model argument allows you to select what pretrained model to use.
             Options include `roberta-base`, `roberta-large`, `allenai/scibert_scivocab_uncased`, `allenai/biomed_roberta_base`, and `PATH TO CHECKPOINT`.
             Note that if you select non-roberta model. You may need to edit the script where it is commented.
* `--dest`: The folder to save weights. The script saves on every epoch.
* `--batch-size-gpu`: The batch size to pass through GPU. You may need to lower this if your GPU has small memory.
* `--batch-size-accumulated`: The accumulated batch size for each optimizer update
* `--lr-base`: The learning rate for the transformer base
* `--lr-linear`: The learning rate for the linear classifier layer.

## Training on SciFact Dataset

Train rationale selection module
```shell script
python verisci/training/rationale_selection/transformer_scifact.py \
    --corpus "PATH TO corpus.jsonl" \
    --claim-train "PATH TO claims_train.jsonl" \
    --claim-dev "PATH TO claims_dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```

For example, after downloading the data using `script/download-data.sh`, you could do
```shell script
python verisci/training/rationale_selection/transformer_scifact.py \
    --corpus "data/corpus.jsonl" \
    --claim-train "data/claims_train.jsonl" \
    --claim-dev "data/claims_dev.jsonl" \
    --model "roberta-large" \
    --dest "model/rationale_roberta_large_scifact"
```

Train label prediction module
```shell script
python verisci/training/label_prediction/transformer_scifact.py \
    --corpus "PATH TO corpus.jsonl" \
    --claim-train "PATH TO claims_train.jsonl" \
    --claim-dev "PATH TO claims_dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```


## Training on FEVER Dataset
You will need to download Fever dataset and Wiki Dump manually
```shell script
# Download Fever dataset:
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
wget https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl

# Download Wikipedia Dump and unzip it manually.
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages.zip
```

Then use the preprocessing script to convert them to our format.
Use it to process both `train.jsonl` and `paper_dev.json` to get output `fever_train.jsonl` and `fever_dev.jsonl`
```shell script
python verisci/training/preprocess_fever.py \
    --wiki-folder "PATH TO wiki-pages folder" \
    --input "PATH TO train.jsonl or paper_dev.jsonl" \
    --output "PATH TO CONVERTED FILE fever_train.jsonl or fever_dev.jsonl"
```

Train rationale selection module
```shell script
python verisci/training/rationale_selection/transformer_fever.py \
    --train "PATH TO fever_train.jsonl" \
    --dev "PATH TO fever_dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```

Train label prediction module
```shell script
python verisci/training/label_prediction/transformer_fever.py \
    --train "PATH TO fever_train.jsonl" \
    --dev "PATH TO fever_dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```


## Training on FEVER + SciFact Dataset
Simply train the model using the Fever script and then using the SciFact script.
When using the SciFact script, change the `--model` argument to the saved weights produced by the
fever script.

## Training on Snopes Dataset
You will need to acquire Snopes dataset yourself.

Train rationale selection module
```shell script
python verisci/training/rationale_selection/transformer_snopes.py \
    --corpus "PATH TO snopes.pages.json" \
    --evidence-train "PATH TO snopes.evidence.train.jsonl" \
    --evidence-dev "PATH TO snopes.evidence.dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```

Train label prediction module
```shell script
python verisci/training/label_prediction/transformer_snopes.py \
    --corpus "PATH TO snopes.pages.json" \
    --evidence-train "PATH TO snopes.stance.train.jsonl" \
    --evidence-dev "PATH TO snopes.stance.dev.jsonl" \
    --model "roberta-large" \
    --dest "PATH TO WEIGHT SAVING FOLDER"
```
