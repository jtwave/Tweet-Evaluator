#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: Lora Config
# * Model: AutoModelForCausalLM GPT-2
# * Evaluation approach: evaluate method with hugging face
# * Fine-tuning dataset: 

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[1]:


from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset

peft_model_name = 'roberta-base-peft'
modified_base = 'roberta-base-modified'
base_model = 'roberta-base'

splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("cardiffnlp/tweet_eval", "emoji", split=splits))}
      
for split in splits:
      ds[split] = ds[split].shuffle(seed = 42).select(range(700))

ds


# In[2]:


model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label= {i: label for i, label in enumerate(ds["train"].features["label"].names)}),
tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding ="max_length", truncation = True)

tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched = True)
    
print(tokenized_ds["train"][0]["input_ids"])


# In[3]:


from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label= {i: label for i, label in enumerate(ds["train"].features["label"].names)})

# Freeze all the parameters of the base model
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.base_model.parameters():
    param.requires_grad = False

model.classifier
print(model)


# In[4]:


import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()


# In[5]:


trainer.evaluate()


# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[6]:


from peft import LoraConfig, get_peft_model
model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label = {i: label for i, label in enumerate(ds["train"].features["label"].names)})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lora_model = get_peft_model(model,config)

lora_model.print_trainable_parameters()


# In[7]:


from transformers import TrainingArguments, Trainer

args=TrainingArguments(
        output_dir="./results",
        # Set the learning rate
        learning_rate = 5e-5,
        # Set the per device train batch size and eval batch size
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        # Evaluate and save the model after each epoch
        evaluation_strategy = "steps",
        save_strategy = "steps",
        num_train_epochs=5,
    )

import numpy as np

import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model = lora_model,
    args = args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator= DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
)
train_results = trainer.train()


# In[10]:


tokenizer.save_pretrained(modified_base)
lora_model.save_pretrained(peft_model_name)


# In[ ]:





# In[ ]:





# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[12]:


from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer


# LOAD the Saved PEFT model

inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, id2label={i: label for i, label in enumerate(ds["train"].features["label"].names)})
tokenizer = AutoTokenizer.from_pretrained(modified_base)


# In[17]:


from torch.utils.data import DataLoader
from tqdm import tqdm
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install evaluate')
import evaluate

metric = evaluate.load('accuracy')

def evaluate_model(inference_model, dataset):

    eval_dataloader = DataLoader(dataset.rename_column("label", "labels"), batch_size=8, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(eval_metric)


# In[27]:


ds = load_dataset("cardiffnlp/tweet_eval", 'emoji')
test_ds=ds.map(preprocess_function, batched=True,  remove_columns=["text"])['test'].shuffle(seed = 42).select(range(200)).shard(num_shards=2, index=1)
# Evaluate the non fine-tuned model
evaluate_model(AutoModelForSequenceClassification.from_pretrained(base_model, id2label={i: label for i, label in enumerate(ds["train"].features["label"].names)}), test_ds)
# Evaluate the PEFT fine-tuned model
evaluate_model(inference_model, test_ds)
# Evaluate the Fully fine-tuned model
evaluate_model(trainer.model, test_ds)


# In[ ]:





# In[ ]:




