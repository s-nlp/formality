from sklearn.metrics import precision_recall_fscore_support, accuracy_score,confusion_matrix, classification_report
import sklearn
import json

from transformers import (DebertaTokenizer,
                          DebertaForSequenceClassification,
                          AutoModelForSequenceClassification,  
                          AutoTokenizer,
                          Trainer,TrainingArguments)
import shutil
import os
from glob import glob
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EarlyStoppingCallback
import gc

from mt5_utils import MT5ForSequenceClassification


def compute_metrics(pred):
    """
    Compute metrics for Trainer
    """
    labels = pred.label_ids
    try:
        preds = pred.predictions.argmax(-1)
    except AttributeError:
        print("Extracting first element of predicts")
        preds = pred.predictions[0]
        print(preds[:10])
        preds = preds.argmax(-1)
        
    #_, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        #'macro f1': macro_f1,
        'precision': precision,
        'recall': recall
    }


def train_nli(datasets, model_type, batch=16, epochs=5, warmup_steps=200, weight_decay = 0.01, lr = 1e-5,save_folder = "/trained_models",
             save_eval_steps=500):
    """
    This contains everything that must be done to train our models
    """
    
    if "mt5" in model_type:
        model = MT5ForSequenceClassification.from_pretrained(model_type, num_labels = 2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels = 2)
        
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    train_dataset, val_dataset, test_dataset = datasets
    
    model_type_save = re.sub("/","_",model_type)
    
    save_folder = f"./{save_folder}/{model_type_save}_ep{epochs}_wus{warmup_steps}_lr{lr}_batch{batch}"
    report_name = f"./{model_type_save}_ep{epochs}_wus{warmup_steps}_lr{lr}_batch{batch}"

    training_args = TrainingArguments(
        output_dir=save_folder,          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=batch,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        # logging_steps=1000,
        eval_steps = save_eval_steps,
        save_steps=save_eval_steps,
        evaluation_strategy = 'steps',
        load_best_model_at_end=True,
        save_strategy="steps", 
        metric_for_best_model="f1",
        greater_is_better = True,
        save_total_limit = 1,
        learning_rate = lr, #: float = 5e-05 default lr from docs,
        report_to = "wandb",
        run_name=report_name,
    )
    
    # training_args.set_save(strategy="epoch", steps=1)

    results = []

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # trainer.place_model_on_device = False
    trainer.train()

    trainer.save_model(f"{save_folder}/nli_model/")
    tokenizer.save_pretrained(f"{save_folder}/nli_model/")    
    
    gc.collect()
    torch.cuda.empty_cache();
    
    