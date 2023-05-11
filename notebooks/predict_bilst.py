from data import load_dataset
from char_bilstm_utils import *
import os
import json
from transformers import (Trainer,TrainingArguments)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",'-m')
parser.add_argument('-dataset', default = "gyafc") #fr pt en it
parser.add_argument('-train_batch', default = 512, type=int)
args = parser.parse_args()

lang = args.model_dir.split(os.path.sep)[-1]
save_folder = f"./get_train_stat/test_results_{lang}"

dataset_dict = load_dataset(model_name=None, dataset_type=args.dataset, language = lang,
             toy=False, test_only = False, get_raw_data = True)

char_vocab = getchar(dataset_dict["train"]["text"])

current_test_dataset = SentenceDataset(dataset_dict["test"]["text"], dataset_dict["test"]["labels"], char_vocab)

existing_inference_results = os.listdir(save_folder)

for model_folder in os.listdir(args.model_dir) :

    if f"{model_folder}.json" not in existing_inference_results and model_folder != ".ipynb_checkpoints":

        print("current model - ", model_folder)

        model_folder_lst = model_folder.split("_")
        ed = int(model_folder_lst[-2][2:])
        hd = int(model_folder_lst[-1][2:])

        model_folder_abs = os.path.join(args.model_dir, model_folder, "nli_model", "pytorch_model.bin")

        #print(os.listdir(model_folder_abs))

        test_model = BiLSTMSequenceClassification(char_vocab=char_vocab, embedding_dim=ed, hidden_dim=hd)
        test_model.load_state_dict(torch.load(model_folder_abs))

        training_args = TrainingArguments(per_device_eval_batch_size=64, output_dir = "./tmp_trainer/", dataloader_drop_last = False)

        trainer = Trainer(model=test_model, args = training_args)

        print("total_test_samples", len(current_test_dataset.labels))

        trainer_preds = trainer.predict(current_test_dataset).predictions

        print(trainer_preds[:10])

        list_predited_label = torch.max(torch.tensor(trainer_preds), dim=1).indices.tolist()
        print("total_predicts", len(list_predited_label))

        with open(f"{save_folder}/{model_folder}.json", "w") as f:
                json.dump(list_predited_label, f)
    else:
        print("dropped", model_folder)