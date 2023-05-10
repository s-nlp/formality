import argparse

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

parser = argparse.ArgumentParser()

parser.add_argument('-model', default = "microsoft/deberta-base")  
parser.add_argument('-train_batch', default = 16, type=int)  

# parser.add_argument('filename')           # positional argument
# parser.add_argument('-c', '--count')      # option that takes a value
# parser.add_argument('-v', '--verbose',
#                     action='store_true')  # on/off flag
args = parser.parse_args()




from data import load_gyafc


tr_val_test_datasets = load_gyafc(args.model, toy = True)
test_ds = tr_val_test_datasets[-1]


test_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = 2)
training_args = TrainingArguments(per_device_eval_batch_size=64, output_dir = "./tmp_trainer/", dataloader_drop_last = True)
                        
trainer = Trainer(model=test_model, args = training_args)
trainer_preds = trainer.predict(test_ds).predictions[0]

print("trainer_preds", trainer_preds[:10])

preds = trainer_preds.argmax(-1)

print("preds", preds[:10])

