## Importing the Libraries and loading the dataset ##

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from huggingface_hub import notebook_login, HfFolder, HfApi
from collections import Counter
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## Performing task specific knowledge distillation using SST-2 dataset - binary classification ##
raw_datasets = load_dataset('glue', 'sst2')

## Checking if GPU is available ##
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

## Name for the repository on the huggingface hub ##
repo_name = "bert-tiny-sst2-KD-BERT"

## Teacher model: https://huggingface.co/gokuls/bert-base-sst2 ##
student_id = "google/bert_uncased_L-2_H-128_A-2" ## using bert-tiny model
teacher_id = "textattack/bert-base-sst2" ## Our pre-trained BERT model is used as teacher

## Checking if the tokenizers of teacher and student model produces the same output ##
# tokenizer initialization #
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
student_tokenizer = AutoTokenizer.from_pretrained(student_id)

# sample input #
sample = "Testing if the tokenizers produce the same output."

# Sanity check #
# assert results
assert teacher_tokenizer(sample) == student_tokenizer(sample), "Tokenizers haven't created the same output"

tokenizer = AutoTokenizer.from_pretrained(teacher_id)

## Tokenization ##
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

## Data Pre-processing ##
tokenized_datasets = tokenized_datasets.remove_columns(['sentence','idx']) ## removing unwanted columns ##
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

## Distillation trainer ##
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # student and teacher on same device #
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output #
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output #
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size #
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss #
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss #
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

## create label2id, id2label dicts ##
labels = tokenized_datasets["train"].features["labels"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

## training args ##
training_args = DistillationTrainingArguments(
    output_dir=repo_name,
    num_train_epochs=50,
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
    # distilation parameters #
    alpha=0.5,
    temperature=3.0
    )

## data_collator ##
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Teacher model #
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

## Student model ##
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

## Evaluation metric - Accuracy ##
def compute_metrics(eval_preds):
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

## Trainer ##
trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
)

## Training ##
trainer.train()

## After the training the Best model will be used ##
## Evaluate ##
trainer.evaluate()

## Saving the model on the hugging face hub ##
## save best model, metrics and create model card ##
trainer.create_model_card(model_name=training_args.hub_model_id)
trainer.push_to_hub()

## Link for the model webpage ##
whoami = HfApi().whoami()
username = whoami['name']
print(f"Model webpage link: https://huggingface.co/{username}/{repo_name}")
