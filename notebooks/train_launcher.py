from utils import train_nli
from data import load_gyafc

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-model', default = "microsoft/deberta-base")  

# parser.add_argument('filename')           # positional argument
# parser.add_argument('-c', '--count')      # option that takes a value
# parser.add_argument('-v', '--verbose',
#                     action='store_true')  # on/off flag
args = parser.parse_args()

tr_val_test_datasets = load_gyafc(args.model, toy = False)

steps_done = 0

for warmup_steps in [2000, 5000, 10000]: #[100, 1000, 2000]
    for epochs in [5]: #10
        for lr in [1e-5, 1e-6]:  # [1e-4, 1e-5, 5e-5, 1e-6]
            
            steps_done += 1
            
            # if steps_done <= 3: continue
            
            train_nli(tr_val_test_datasets, 
                      model_type = args.model, 
                      epochs=epochs, 
                      warmup_steps=warmup_steps, 
                      lr=lr)
            
            










