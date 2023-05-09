from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import shutil
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import os 

# Gyafc

def data_read(data_path):
    data = []
    for file_name in glob(data_path):
        with open(file_name) as f:
            tmp_data = f.read()
            data.extend(tmp_data.split('\n'))
    return data

def prep_dataset(formal, informal):
    tuples = []
    data = []
    labels = []
    formal = list(set(formal))
    for sentence in formal:
        data.append(sentence)
        labels.append(0)
    informal = list(set(informal))
    for sentence in informal:
        data.append(sentence)
        labels.append(1)
    return data, labels

class Formal_informal(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    


def load_gyafc(model_name, toy=False):
    
    
    dir_path = os.path.dirname(os.path.realpath(__file__))


    path_formal = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/formal*')
    path_inform = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/informal*')
    
    data_train_form = data_read(path_formal.format('train'))
    data_train_inform = data_read(path_inform.format('train'))

    data_valid_form = data_read(path_formal.format('test'))
    data_valid_inform = data_read(path_inform.format('test'))

    data_test_form = data_read(path_formal.format('tune'))
    data_test_inform = data_read(path_inform.format('tune'))
    
    if toy == True:
        data_train_form = data_train_form[:100]
        data_train_inform = data_train_inform[:100]
        data_valid_form = data_valid_form[:100]
        data_valid_inform = data_valid_inform[:100]
        data_test_form = data_test_form[:100]
        data_test_inform = data_test_inform[:100]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_texts, train_labels = prep_dataset(data_train_form, data_train_inform)
    val_texts, val_labels = prep_dataset(data_valid_form, data_valid_inform)
    test_texts, test_labels = prep_dataset(data_test_form, data_test_inform)
            
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=24)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=24)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=24)
    
    train_dataset = Formal_informal(train_encodings, train_labels)
    val_dataset = Formal_informal(val_encodings, val_labels)
    test_dataset = Formal_informal(test_encodings, test_labels)
    
    return (train_dataset, val_dataset, test_dataset)

    # return {"train":{"encodings":train_encodings, "labels":train_labels},
    #        "val":{"encodings":val_encodings, "labels":val_labels},
    #        "test":{"encodings":test_encodings, "labels":test_labels}}
    
    

class SentenceDataset(Dataset):
    def __init__(self, datasets):
      self.formal = []
      self.informal = []
      if isinstance(datasets, list):
        for dataset in datasets:
            self.formal += [sentence for sentence in dataset['formal']]
            self.informal += [sentence for sentence in dataset['informal']]  
      else:
        for language in datasets:
          self.formal += [sentence for sentence in datasets[language]['formal']]
          self.informal += [sentence for sentence in datasets[language]['informal']]        

      print(f"formal sentences: {len(self.formal)}")
      print(f"informal sentences: {len(self.informal)}")

      self.tuples = []
      self.tuples += [(sentence,0) for sentence in self.formal] 
      self.tuples += [(sentence,1) for sentence in self.informal]
      shuffle(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]
    
def get_files(path_dataset,):
    
    print(path_dataset)


    data = {"fr" : {"formal" : [], "informal" : []},
            "pt" : {"formal" : [], "informal" : []},
            "en" : {"formal" : [], "informal" : []},
            "it" : {"formal" : [], "informal" : []},
            "ru" : {"formal" : [], "informal" : []}
            }

    for file_name in glob(path_dataset):
      # print(file_name)
      with open(file_name, "r") as f:
        content = f.readlines()

        data[file_name[25:27]][str(get_label(file_name).numpy())[2:-1]] += content


    data = {
            "fr" : {"formal": [sentence for sentence in list(set(data["fr"]["formal"])) if len(sentence) <=150],
                    "informal": [sentence for sentence in list(set(data["fr"]["informal"])) if len(sentence) <=150]},
            
            "pt" : {"formal": [sentence for sentence in list(set(data["pt"]["formal"])) if len(sentence) <=150],
                    "informal": [sentence for sentence in list(set(data["pt"]["informal"])) if len(sentence) <=150]},
            
            "en" : {"formal": [sentence for sentence in list(set(data["en"]["formal"])) if len(sentence) <=150],
                    "informal": [sentence for sentence in list(set(data["en"]["informal"])) if len(sentence) <=150]},
            
            "it" : {"formal": [sentence for sentence in list(set(data["it"]["formal"])) if len(sentence) <=150],
                    "informal": [sentence for sentence in list(set(data["it"]["informal"])) if len(sentence) <=150]}, 
            # "ru" : {"formal": [sentence for sentence in list(set(data["ru"]["formal"])) if len(sentence) <=150],
            #         "informal": [sentence for sentence in list(set(data["ru"]["informal"])) if len(sentence) <=150]}, 
            }
            

    return data

def load_xformal(model_name, language, toy=False):
    
    
    train_data  = get_files(data_path.format('train'))
    validation_data = get_files(data_path.format('test'))
    test_data = get_files(data_path.format('tune'))
    
    if language == "all":
      train_dataset  = SentenceDataset([train_data[temp] for temp in train_data.keys()])
      val_dataset  = SentenceDataset([validation_data[temp] for temp in validation_data.keys()])
      test_dataset  = SentenceDataset([test_data[temp] for temp in test_data.keys()])