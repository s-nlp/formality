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

train_nli(tr_val_test_datasets, 
          model_type = args.model, 
          epochs=10, 
          warmup_steps=200, 
          weight_decay = 0.01)










