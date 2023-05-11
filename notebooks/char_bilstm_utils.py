from torch import nn
from torch.utils.data import Dataset
from transformers import MT5PreTrainedModel, MT5Model, MT5Config
from transformers.models.mt5 import MT5Stack
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch
from torch import nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy

class SentenceDataset(Dataset):
    def __init__(self, texts, labels, char_vocab):

        self.encodings = {"input_ids":[text_pipeline(txt, char_vocab) for txt in texts]}
        self.labels =  labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

from torchtext.vocab import vocab
def getchar(text):
    PRUNE_TOKENS_LESS_THAN = 0

    tokens = set.union(*[set(sentence) for sentence in text])
    token_stats = {token :  0 for token in tokens}
    text = ' '.join(text)
    for token in list(tokens):
      token_stats[token] = text.count(token)

    tokens = [key for (key,value) in token_stats.items() if value >= PRUNE_TOKENS_LESS_THAN]
    tokens = sorted(tokens)
    tokens = {key:id for (id,key) in enumerate(tokens)}
    tokens["<EMP>"] = len(tokens)
    tokens["<UNK>"] = len(tokens)

    char_vocab = vocab(tokens)
    char_vocab.set_default_index(len(tokens)-1)

    return char_vocab

def text_pipeline(text, char_vocab, length=64):
  #print(text)
  chars = [char for char in text]
  text = char_vocab(chars)
  
  if len(text) > length:
    text = text[:length]
  else:
    text += [char_vocab["<EMP>"]] * (length - len(text))

  #print(text)
  #print(len(text))
  #raise Exception("STOP")
  return torch.tensor(text, dtype=torch.int64)

from transformers import PreTrainedModel, PretrainedConfig

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BiLSTMSequenceClassification(nn.Module):  #PreTrainedModel

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ with T5->MT5
    def __init__(self, char_vocab, embedding_dim=50, hidden_dim=50, lstm_layer=2, output=2):
        super().__init__()

        self.config = PretrainedConfig()
        self.num_labels = output

        self.hidden_dim = hidden_dim

        # load pre-trained embeddings
        self.embedding = nn.Embedding(len(char_vocab)+1, embedding_dim)
        # embeddings are not fine-tuned
        #self.embedding.weight.requires_grad = False

        # RNN layer with LSTM cells
        # OR self.lstm = NaiveLSTM(input_sz = self.embedding.embedding_dim, hidden_sz = hidden_dim)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            bidirectional=True,
                            dropout = 0.5)

        self.pooler = Pooler(hidden_dim*2)

        self.output = nn.Linear(hidden_dim*2, output)
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None
    ) :

        #print("input_ids", input_ids.shape)

        outputs = self.embedding(input_ids)

        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(outputs)

        #print("lstm_out",lstm_out.shape)

        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        #lstm_out = lstm_out.view(x.shape[0], -1, 2, self.hidden_dim)#.squeeze(1)
        #print("lstm_out view",lstm_out.shape)

        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        #dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        # I drop this line of code because it breaks dimensions and does not seem to make sens

        #print("dense_input", dense_input.shape)

        pooled = self.pooler(lstm_out)
        #print("pooled", pooled.shape)

        logits = self.output(pooled)#.view([1, 2])

        #print("logits", logits.shape)
        #print("labels", labels.shape)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )

class BiLSTM(nn.Module):
    def __init__(self, char_vocab, embedding_dim=50, hidden_dim=50, lstm_layer=2, output=2):

        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # load pre-trained embeddings
        self.embedding = nn.Embedding(len(char_vocab)+1, embedding_dim)
        # embeddings are not fine-tuned
        #self.embedding.weight.requires_grad = False

        # RNN layer with LSTM cells
        # OR self.lstm = NaiveLSTM(input_sz = self.embedding.embedding_dim, hidden_sz = hidden_dim)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            bidirectional=True,
                            dropout = 0.5)
        # dense layer
        self.output = nn.Linear(hidden_dim*2, output)

    def forward(self, sents):
        x = self.embedding(sents)

        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)

        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        lstm_out = lstm_out.view(x.shape[0], -1, 2, self.hidden_dim)#.squeeze(1)

        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)

        y = self.output(dense_input)#.view([1, 2])
        return y



def train(dataloader, log_interval):
    model.train()
    final_loss, accuracy, count = 0, 0, 0
    interval_time = time.time()
    epoch_time = interval_time
    for idx, (inputs, outputs) in enumerate(dataloader):
      cleanup()
      optimizer.zero_grad()

      inputs = [input.to(device) for input in inputs]
      outputs = [output.to(device) for output in outputs]

      predicted_outputs = model(inputs)

      for output, predicted_output in zip(outputs, predicted_outputs):

        loss = criterion(predicted_output, output)
        loss.backward()
        final_loss += loss.item()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
      optimizer.step()

      accuracy += (predicted_outputs[0].argmax(1) == outputs[0]).sum().item()
      count += outputs[0].size(0)

      if idx % log_interval == 0 and idx > 0:
        elapsed = time.time() - interval_time
        print(f'| epoch {epoch} \| {idx}/{len(dataloader)} batches \| accuracy {accuracy/count} \| time elapsed {elapsed} \| batches/second {log_interval/elapsed}')
        interval_time = time.time()



    cleanup()
    return  final_loss/count, accuracy/count, time.time() -epoch_time



def evaluate(dataloader):
  model.eval()
  accuracy, loss, count = 0,0,0
  with torch.no_grad():
    for idx, (inputs, outputs) in enumerate(dataloader):
      cleanup()

      inputs = [input.to(device) for input in inputs]
      outputs = [output.to(device) for output in outputs]

      predicted_outputs = model(inputs)

      for output, predicted_output in zip(outputs, predicted_outputs):
        loss += criterion(predicted_output, output)

      accuracy += (predicted_outputs[0].argmax(1) == outputs[0]).sum().item()
      count += outputs[0].size(0)

      cleanup()
  return loss/count, accuracy/count

