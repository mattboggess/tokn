import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class MaxPoolBert(BaseModel):
    
    def __init__(self, num_classes):
        super().__init__()
        #TODO: Add dropout
        self.bert = BertModel.from_pretrained("bert-base-cased")
        bert_config = self.bert.config.__dict__
        self.fc_e = nn.Linear(bert_config["hidden_size"], bert_config["hidden_size"])
        self.fc_cls = nn.Linear(bert_config["hidden_size"], bert_config["hidden_size"])
        self.fc_class = nn.Linear(bert_config["hidden_size"] * 3, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, batch_data):
        
        x = batch_data["data"]
        pad_mask = batch_data["pad_mask"]
        e1_mask = batch_data["e1_mask"]
        e2_mask = batch_data["e2_mask"]
        sentence_mask = batch_data["sentence_mask"]
        
        batch_size = x.shape[0]
        num_sentences = x.shape[1]
        seq_length = x.shape[2]
        
        # reshape to collapse batch size and num_sentences for bert input
        x = x.view(batch_size * num_sentences, seq_length)
        pad_mask = pad_mask.view(batch_size * num_sentences, seq_length)
        
        # pass through bert
        bert_output = self.bert(x, attention_mask=pad_mask)
        
        # get sentence level output
        cls_output = bert_output[1]
        cls_output = cls_output.view(batch_size, num_sentences, cls_output.shape[-1])
        
        # compute e1 and e2 averages
        full_output = bert_output[0]
        full_output = full_output.view(batch_size, num_sentences, full_output.shape[-2], 
                                       full_output.shape[-1])
        e1_mask = e1_mask.unsqueeze(-1)
        e2_mask = e2_mask.unsqueeze(-1)
        e1_avg = (full_output * e1_mask).mean(dim=-2)
        e2_avg = (full_output * e2_mask).mean(dim=-2)
        
        # pass through fully connected layers w/ activation functions
        cls_output = self.fc_cls(cls_output).tanh()
        e1_avg = self.fc_e(e1_avg).tanh()
        e2_avg = self.fc_e(e2_avg).tanh()
        
        # concatenate sentence level, e1, and e2
        rel_rep = torch.cat([cls_output, e1_avg, e2_avg], dim=-1)
        
        # max "pool" across sentences (mask out padding sentences)
        rel_rep += ((1 - sentence_mask.unsqueeze(-1)) * -10**10)
        rel_rep = torch.max(rel_rep, dim=-2)[0]
        
        # fully connected + softmax for classification
        output = self.fc_class(rel_rep)
        output = self.softmax(output)
        
        return output

class MaxPoolTest(BaseModel):
    """
    Arbritrary baseline model that collapses LSTM output with random word embeddings into
    single vector that can be max-pooled across sentences. Not Legit. Used for pipeline dev.
    """
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_classes, max_sent_length):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size * max_sent_length, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, pad_mask, e1_mask, e2_mask):
        x = self.embeddings(x)
        old_size = x.size()
        x = x.view(-1, old_size[-2], old_size[-1])
        x = self.lstm(x)[0]
        x = x.view(old_size[0], old_size[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.max(x, dim=-1)[0]
        x = x.squeeze(-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
