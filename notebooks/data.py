from transformers import AutoTokenizer
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import os
# import tensorflow as tf

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

def unroll_to_two_lists(data_dict, languages):
    formal_list = []
    informal_list = []

    for lang in languages:
        formal_list.extend(data_dict[lang]["formal"])
        informal_list.extend(data_dict[lang]["informal"])

    return formal_list, informal_list

def load_dataset(model_name, dataset_type="gyafc", language = None, toy=False):
    
    
    dir_path = os.path.dirname(os.path.realpath(__file__))


    if dataset_type == "gyafc":

        path_formal = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/formal*')
        path_inform = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/informal*')

        data_train_form = data_read(path_formal.format('train'))
        data_train_inform = data_read(path_inform.format('train'))

        data_valid_form = data_read(path_formal.format('test'))
        data_valid_inform = data_read(path_inform.format('test'))

        data_test_form = data_read(path_formal.format('tune'))
        data_test_inform = data_read(path_inform.format('tune'))

    elif dataset_type == "xformal":

        # data_path =  os.path.join(dir_path,'XFORMAL/gyafc_translated/*/*/{}/*/*')
        data_path = 'XFORMAL/gyafc_translated/*/*/{}/*/*'

        train_data = get_files(data_path.format('train'))
        validation_data = get_files(data_path.format('test'))
        test_data = get_files(data_path.format('tune'))

        if language == "all":
            data_train_form, data_train_inform = unroll_to_two_lists(train_data, list(train_data.keys()))
            data_valid_form, data_valid_inform = unroll_to_two_lists(validation_data, list(validation_data.keys()))
            data_test_form, data_test_inform = unroll_to_two_lists(test_data, list(test_data.keys()))

        elif "only" in language:
            only_lang = language.split("_")[0]
            print(f"Will be trained only on {only_lang}")
            data_train_form, data_train_inform = unroll_to_two_lists(train_data, [only_lang])
            data_valid_form, data_valid_inform = unroll_to_two_lists(validation_data, [only_lang])

            if only_lang == "en":
                path_formal = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/formal*')
                path_inform = os.path.join(dir_path, 'GYAFC_Corpus/*/{}/informal*')
                data_test_form = data_read(path_formal.format('tune'))
                data_test_inform = data_read(path_inform.format('tune'))
            else:
                data_test_form, data_test_inform = unroll_to_two_lists(test_data, [only_lang])

        elif "all_but" in language:
            all_but_lang = language.split("_")[-1]
            print(f"Will be trained on all but {all_but_lang}")
            data_train_form, data_train_inform = unroll_to_two_lists(train_data, [lang for lang in list(train_data.keys()) if lang!= all_but_lang])
            data_valid_form, data_valid_inform = unroll_to_two_lists(validation_data, [lang for lang in list(train_data.keys()) if lang!= all_but_lang])
            data_test_form, data_test_inform = unroll_to_two_lists(test_data, [all_but_lang])

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

# def get_label(file_path):
#     parts = tf.strings.split(file_path, os.path.sep)
#     # Note: You'll use indexing here instead of tuple unpacking to enable this
#     # to work in a TensorFlow graph.
#     return parts[-2]

def get_label(file_path):
    # print("file_path",file_path)
    parts = file_path.split(os.path.sep)
    # print("parts",parts)
    return parts[-1].split(".")[0], parts[-5]


def get_files(path_dataset, ):
    data = {"fr": {"formal": [], "informal": []},
            "pt": {"formal": [], "informal": []},
            "en": {"formal": [], "informal": []},
            "it": {"formal": [], "informal": []},
            }

    for file_name in glob(path_dataset):
        # print(file_name)
        with open(file_name, "r") as f:
            content = f.readlines()

            label, lang = get_label(file_name)
            #         print(label, lang)
            if lang != "ru":
                data[lang][label] += content

    data = {
        "fr": {"formal": [sentence for sentence in list(set(data["fr"]["formal"])) if len(sentence) <= 150],
               "informal": [sentence for sentence in list(set(data["fr"]["informal"])) if len(sentence) <= 150]},

        "pt": {"formal": [sentence for sentence in list(set(data["pt"]["formal"])) if len(sentence) <= 150],
               "informal": [sentence for sentence in list(set(data["pt"]["informal"])) if len(sentence) <= 150]},

        "en": {"formal": [sentence for sentence in list(set(data["en"]["formal"])) if len(sentence) <= 150],
               "informal": [sentence for sentence in list(set(data["en"]["informal"])) if len(sentence) <= 150]},

        "it": {"formal": [sentence for sentence in list(set(data["it"]["formal"])) if len(sentence) <= 150],
               "informal": [sentence for sentence in list(set(data["it"]["informal"])) if len(sentence) <= 150]},
    }

    return data