# Knowledge-Distillation
* Knowledge Distillation is a model-agnostic technique to compresses and transfers the knowledge from a computationally expensive large deep neural network(Teacher) to a single smaller neural work(Student) with better inference efficiency.
* ![]()
* In this example, we will use a BERT-base as Teacher and BERT-Tiny as Student.
* We will use Text-Classification as our task-specific knowledge distillation task and the Stanford Sentiment Treebank v2 (SST-2) dataset for training.
* They are two different types of knowledge distillation, the Task-agnostic knowledge distillation (right) and the Task-specific knowledge distillation (left). In this example we are going to use the Task-specific knowledge distillation.
* ![](community-score.png)
* In Task-specific knowledge distillation a "second step of distillation" is used to "fine-tune" the model on a given dataset. This idea comes from the DistilBERT paper where it was shown that a student performed better than simply finetuning the distilled language model.

