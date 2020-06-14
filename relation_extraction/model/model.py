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

class BertEM(BaseModel):
    """
    BERT-EM model as proposed here: https://arxiv.org/pdf/1906.03158v1.pdf
    Modified version of BERT for relation extraction.
    """
    def __init__(self, num_classes, vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.resize_token_embeddings(vocab_size)
        bert_config = self.bert.config.__dict__
        self.fc_relation = nn.Linear(bert_config['hidden_size'] * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
        #self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc_relation.weight)

    def forward(self, batch_data):
        
        x = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        term1_mask = batch_data['term1_mask']
        term2_mask = batch_data['term2_mask']
        
        # pass through bert
        bert_output = self.bert(x, attention_mask=attention_mask)
        bert_output = bert_output[0]
        
        # concatenate term mask representations
        term1_rep = (bert_output * term1_mask.unsqueeze(-1)).sum(dim=-2)
        term2_rep = (bert_output * term2_mask.unsqueeze(-1)).sum(dim=-2)
        term_pair_rep = torch.cat([term1_rep, term2_rep], dim=-1)
        
        # generate logsoftmax output
        output = self.fc_relation(term_pair_rep)
        output = self.softmax(output)
        
        return output
    
