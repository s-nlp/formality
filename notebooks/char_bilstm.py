from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import time
import math

from data import load_dataset
from math import floor

from char_bilstm_utils import *
from utils import compute_metrics

from sklearn.metrics import precision_recall_fscore_support, accuracy_score,confusion_matrix, classification_report
import sklearn
import json
import wandb

from transformers import Trainer,TrainingArguments
import shutil
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EarlyStoppingCallback

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default = "xformal") #fr pt en it
parser.add_argument('-language',default = "en_only")
parser.add_argument('-train_batch', default = 512, type=int)
args = parser.parse_args()

import gc
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

dataset_dict = load_dataset(model_name=None, dataset_type=args.dataset, language = args.language,
             toy=False, test_only = False, get_raw_data = True)

char_vocab = getchar(dataset_dict["train"]["text"])

train_dataset = SentenceDataset(dataset_dict["train"]["text"], dataset_dict["train"]["labels"], char_vocab)
val_dataset = SentenceDataset(dataset_dict["val"]["text"], dataset_dict["val"]["labels"], char_vocab)
test_dataset = SentenceDataset(dataset_dict["test"]["text"], dataset_dict["test"]["labels"], char_vocab)

for lr in [1e-5, 1e-6]:
    for embedding_dim in [50,100]:
        for hidden_dim in [50,100]:

            model = BiLSTMSequenceClassification(char_vocab=char_vocab,
                                                 embedding_dim=embedding_dim,
                                                  hidden_dim=hidden_dim)

            report_name = f"bilstm_ds_{args.dataset}_lang_{args.language}_lr{lr}_ed{embedding_dim}_hd{hidden_dim}"
            save_folder =f"./trained_blstms/{args.language}/{report_name}"

            gpus = torch.cuda.device_count()

            warmup_steps = int((len(dataset_dict["train"]["text"])/args.train_batch)*0.5/gpus)
            save_eval_steps = int((len(dataset_dict["train"]["text"])/args.train_batch)*0.5/gpus)

            training_args = TrainingArguments(
                output_dir=save_folder,          # output directory
                num_train_epochs=5,              # total number of training epochs
                per_device_train_batch_size=args.train_batch,  # batch size per device during training
                per_device_eval_batch_size=args.train_batch,   # batch size for evaluation
                warmup_steps=100,                # number of warmup steps for learning rate scheduler
                weight_decay=0.01,               # strength of weight decay
                logging_dir='./logs',            # directory for storing logs
                logging_steps=100,
                eval_steps = save_eval_steps,
                save_steps=save_eval_steps,
                evaluation_strategy = 'steps',
                load_best_model_at_end=True,
                save_strategy="steps",
                metric_for_best_model="f1",
                greater_is_better = True,
                save_total_limit = 1,
                learning_rate = lr, #: float = 5e-05 default lr from docs
                #report_to="none",
                report_to = "wandb",
                run_name=report_name,
            )

            # training_args.set_save(strategy="epoch", steps=1)

            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=val_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
            )
            #trainer.place_model_on_device = False
            trainer.train()

            trainer.save_model(f"{save_folder}/nli_model/")

            gc.collect()
            torch.cuda.empty_cache()

            wandb.finish()






