import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np
import torch
from transformers import BertModel
from torchcrf import CRF

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
        #nn.init.xavier_normal_(self.fc.weight) # TODO
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, batch_data):
        s, _ = self.bert(batch_data["data"], attention_mask=batch_data["pad_mask"])
        s = self.dropout(s)
        s = self.fc(s)
        s = self.softmax(s)
        return s

class BertCRFNER(BaseModel):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    
    def __init__(self, num_classes, dropout_rate=0.3, tags=["O", "S", "B", "I", "E"]):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        bert_config = self.bert.config.__dict__
        
        self.fc = nn.Linear(bert_config["hidden_size"], num_classes)
        #nn.init.xavier_normal_(self.fc.weight) # TODO
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.crf = CRF(len(tags), batch_first=True)
        
        # Can't start with interior/end of term phrase
        #self.crf.start_transitions.data[tags.index("I")] = -1e5
        #self.crf.start_transitions.data[tags.index("E")] = -1e5
        
        # Can't end with interior/beginning of term phrase
        #self.crf.end_transitions.data[tags.index("I")] = -1e5
        #self.crf.end_transitions.data[tags.index("B")] = -1e5
        
        # Unlikely to immediately transition to another term 
        self.crf.transitions.data[tags.index("E"), tags.index("B")] = -1e5
        self.crf.transitions.data[tags.index("E"), tags.index("S")] = -1e5
        self.crf.transitions.data[tags.index("S"), tags.index("B")] = -1e5
        self.crf.transitions.data[tags.index("S"), tags.index("S")] = -1e5
        
        # Must form valid term phrase 
        self.crf.transitions.data[tags.index("B"), tags.index("O")] = -1e5
        self.crf.transitions.data[tags.index("B"), tags.index("S")] = -1e5
        self.crf.transitions.data[tags.index("B"), tags.index("B")] = -1e5
        self.crf.transitions.data[tags.index("I"), tags.index("O")] = -1e5
        self.crf.transitions.data[tags.index("I"), tags.index("S")] = -1e5
        self.crf.transitions.data[tags.index("I"), tags.index("B")] = -1e5
        self.crf.transitions.data[tags.index("E"), tags.index("I")] = -1e5
        self.crf.transitions.data[tags.index("E"), tags.index("E")] = -1e5
        self.crf.transitions.data[tags.index("O"), tags.index("I")] = -1e5
        self.crf.transitions.data[tags.index("O"), tags.index("E")] = -1e5
        self.crf.transitions.data[tags.index("S"), tags.index("I")] = -1e5
        self.crf.transitions.data[tags.index("S"), tags.index("E")] = -1e5
        
    def forward(self, batch_data):
        s, _ = self.bert(batch_data["data"], attention_mask=batch_data["pad_mask"])
        s = self.dropout(s)
        emissions = self.fc(s)
        s = self.softmax(s)
        return emissions 

    def decode(self, emissions, mask):
        emissions = emissions[:, 1:, :]
        mask = mask.to(torch.uint8)[:, 1:]
        return self.crf.decode(emissions)

