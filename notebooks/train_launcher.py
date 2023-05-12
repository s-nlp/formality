from utils import train_nli
from data import load_dataset
import torch
import os
import argparse

# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

parser = argparse.ArgumentParser()


parser.add_argument('-model', default = "distilbert-base-multilingual-cased")
parser.add_argument('-dataset', default = "xformal") #fr pt en it
parser.add_argument('-language', required=True)

parser.add_argument('-train_batch', default = 16, type=int)  

# parser.add_argument('filename')           # positional argument
# parser.add_argument('-c', '--count')      # option that takes a value
# parser.add_argument('-v', '--verbose',
#                     action='store_true')  # on/off flag
args = parser.parse_args()

tr_val_test_datasets = load_dataset(args.model, dataset_type =args.dataset, language = args.language, toy = False)
train_ds = tr_val_test_datasets[0]

steps_done = 0

gpus_number = torch.cuda.device_count()

for batch in [args.train_batch]:
    for warmup_steps_frac in [0.5]: #fraction from epoch
        
        warmup_steps = int((len(train_ds)/batch)*warmup_steps_frac/gpus_number)
        save_eval_steps = int((len(train_ds)/batch)*0.5/gpus_number)

        for epochs in [5]: #10
            for lr in [5e-5]:  # [1e-4, 1e-5, 5e-5, 1e-6]

                steps_done += 1
                #if steps_done <= 1: continue
                if steps_done > 1: break

                save_folder = f"./trained_models_{args.language}"

                if not os.path.exists(save_folder):
                   os.makedirs(save_folder)

                train_nli(datasets=tr_val_test_datasets,
                          batch = batch,
                          model_type = args.model, 
                          epochs=epochs, 
                          warmup_steps=warmup_steps, 
                          save_eval_steps = save_eval_steps,
                          lr=lr,
                          save_folder = save_folder,
                          language=args.language)
            
            











