from transformers import (DebertaTokenizer,
                          DebertaForSequenceClassification,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          Trainer,TrainingArguments)

import torch
import tensorflow as tf

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels = 2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

training_args = TrainingArguments(per_device_eval_batch_size=64, output_dir = "./tmp_trainer/", dataloader_drop_last = True)

trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,
        )

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)