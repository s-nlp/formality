import sys
sys.path.append("../")

from data import load_gyafc
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import (DebertaTokenizer,
                          DebertaForSequenceClassification,
                          AutoModelForSequenceClassification,  
                          AutoTokenizer,
                          Trainer,TrainingArguments)
import os
import json
from tqdm import tqdm


def get_test_dataset(model_name):
    tr_val_test_datasets = load_gyafc(model_name, toy = False)
    _, _, test_dataset = tr_val_test_datasets
    return test_dataset

def get_model_type(model_name):

    if "mbart-large" in model_name:
        return "facebook/mbart-large-50"
    elif "distilbert" in model_name:
        return "distilbert-base-multilingual-cased"
    elif "bert-base-multilingual-cased" in model_name:
        return "bert-base-multilingual-cased"
    elif "bigscience_bloom-1b1" in model_name:
        return "bigscience/bloom-1b1"
    else:
        raise Exception ("Unhandled model type!!")

test_dataset_dict = {}

trained_models_fld = "../trained_models/"

existing_inference_results = os.listdir("test_results_trainer/")

for model_folder in os.listdir(trained_models_fld) :
    
    if f"{model_folder}.json" not in existing_inference_results and model_folder != ".ipynb_checkpoints":
        
        print("current model - ", model_folder)
        
        model_type = get_model_type(model_folder)
        
        if model_type not in test_dataset_dict:
            test_dataset_dict[model_type] = get_test_dataset(model_type)

        current_test_dataset = test_dataset_dict[model_type]

        model_folder_abs = os.path.join(trained_models_fld, model_folder)
        
        #print(os.listdir(model_folder_abs))

        if os.path.isdir(f"{model_folder_abs}/nli_model/")==True:
            test_model = AutoModelForSequenceClassification.from_pretrained(f"{model_folder_abs}/nli_model/")
            print("Model loaded")
        else:
            print("Not trained! Skipping ...")
            continue
        
        training_args = TrainingArguments(per_device_eval_batch_size=64, output_dir = "./tmp_trainer/", dataloader_drop_last = True)
        
        trainer = Trainer(model=test_model, args = training_args)
        
        if "mbart" in model_type:
            trainer_preds = trainer.predict(current_test_dataset).predictions[0]
            trainer_preds = torch.tensor(trainer_preds)
        else:
            trainer_preds = trainer.predict(current_test_dataset).predictions

        print(trainer_preds[:10])

        list_predited_label = torch.max(torch.tensor(trainer_preds), dim=1).indices.tolist()    
        
        with open(f"test_results_trainer/{model_folder}.json", "w") as f:
                json.dump(list_predited_label, f)
    else:
        print("dropped", model_folder)
