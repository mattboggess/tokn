import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np
import torch
from transformers import BertModel

class BaseModel(nn.Module):
    """
    Base class for all models. Wrapper that provides a print-friendly representation of the 
    model parameters.
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

class BertNER(BaseModel):
    
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        bert_config = self.bert.config.__dict__
        self.fc = nn.Linear(bert_config["hidden_size"], num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, batch_data):
        s, _ = self.bert(batch_data["data"], attention_mask=batch_data["pad_mask"])
        s = self.dropout(s)
        s = self.fc(s)
        s = self.softmax(s)
        return s

