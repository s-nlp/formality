from utils import train_nli
from data import load_gyafc

import argparse

# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

parser = argparse.ArgumentParser()

parser.add_argument('-model', default = "microsoft/deberta-base")  #facebook/mbart-large-50 google/mt5-base
parser.add_argument('-train_batch', default = 16, type=int)  

# parser.add_argument('filename')           # positional argument
# parser.add_argument('-c', '--count')      # option that takes a value
# parser.add_argument('-v', '--verbose',
#                     action='store_true')  # on/off flag
args = parser.parse_args()

tr_val_test_datasets = load_gyafc(args.model, toy = False)
train_ds = tr_val_test_datasets[0]

steps_done = 0

for batch in [args.train_batch]:
    for warmup_steps_frac in [0.25, 0.5, 1]: #fraction from epoch
        
        warmup_steps = int((len(train_ds)/batch)*warmup_steps_frac)
        save_eval_steps = int((len(train_ds)/batch)*0.5)

        for epochs in [5]: #10
            for lr in [1e-5, 1e-6]:  # [1e-4, 1e-5, 5e-5, 1e-6]

                steps_done += 1
                if steps_done <= 1: continue

                train_nli(datasets=tr_val_test_datasets,
                          batch = batch,
                          model_type = args.model, 
                          epochs=epochs, 
                          warmup_steps=warmup_steps, 
                          save_eval_steps = save_eval_steps,
                          lr=lr)