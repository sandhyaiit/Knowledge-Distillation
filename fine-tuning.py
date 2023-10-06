## Importing the libraries and loading the dataset ##

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from huggingface_hub import notebook_login, HfFolder, HfApi
from collections import Counter
import evaluate
import numpy as np
import torch
import os

## Using SST-2 dataset - binary classification ##
raw_datasets = load_dataset("glue","sst2")

## Checking if GPU is available ##
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

## Name for the repository on the huggingface hub ##
repo_name = "bert-base-sst2"

## Model used for fine-tuning ##
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

## Tokenization ##
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## Data Pre-processing ##
tokenized_datasets = tokenized_datasets.remove_columns(['sentence','idx']) ## removing unwanted columns ##
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

## Create label2id, id2label dicts - to store id and label values ##
labels = tokenized_datasets["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

## Training the Model ##
training_args = TrainingArguments(checkpoint)
training_args

## Training Arguments ##
training_args = TrainingArguments(
    output_dir=repo_name,
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    learning_rate=5e-5,
    seed=33,
    # logging & evaluation strategies #
    logging_dir=f"{repo_name}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    # push to hub parameters #
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    hub_token=HfFolder.get_token(),
    )

## Evaluation metric ##
def compute_metrics(eval_preds):
    metric_acc = evaluate.load("accuracy") ## Using accuracy as a performance evaluation measure ##
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

## Model ##
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id) ## Number of classes = 2 (positive and negative) ##

## Trainer ##
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)], ## For early stopping (patience = 3) ##
)


## Training ##
trainer.train()

## Evaluate ##
trainer.evaluate()


## Saving the model on the hugging face hub ##

# save best model, metrics and create model card #
trainer.create_model_card(model_name=training_args.hub_model_id)
trainer.push_to_hub()

## Link for the model webpage ##
whoami = HfApi().whoami()
username = whoami['name']
print(f"Model webpage link: https://huggingface.co/{username}/{repo_name}")
