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
        self.softmax = nn.LogSoftmax(dim=-1)
        
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
        e1_avg = (full_output * e1_mask.to(torch.float32)).mean(dim=-2)
        e2_avg = (full_output * e2_mask.to(torch.float32)).mean(dim=-2)
        
        # pass through fully connected layers w/ activation functions
        cls_output = self.fc_cls(cls_output).tanh()
        e1_avg = self.fc_e(e1_avg).tanh()
        e2_avg = self.fc_e(e2_avg).tanh()
        
        # concatenate sentence level, e1, and e2
        rel_rep = torch.cat([cls_output, e1_avg, e2_avg], dim=-1)
        
        # max "pool" across sentences (mask out padding sentences)
        rel_rep += ((1 - sentence_mask.unsqueeze(-1)) * -1e8)
        rel_rep = torch.max(rel_rep, dim=-2)[0]
        
        # fully connected + softmax for classification
        output = self.fc_class(rel_rep)
        output = self.softmax(output)
        
        return output
    
class BagAttentionBert(BaseModel):
    
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        bert_config = self.bert.config.__dict__
        self.fc_e = nn.Linear(bert_config["hidden_size"], bert_config["hidden_size"])
        nn.init.kaiming_uniform_(self.fc_e.weight)
        self.fc_cls = nn.Linear(bert_config["hidden_size"], bert_config["hidden_size"])
        nn.init.kaiming_uniform_(self.fc_cls.weight)
        self.fc_class = nn.Linear(bert_config["hidden_size"] * 3, num_classes)
        nn.init.kaiming_uniform_(self.fc_class.weight)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, batch_data, evaluate=True):
        
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
        e1_avg = (full_output * e1_mask.to(torch.float32)).mean(dim=-2)
        e2_avg = (full_output * e2_mask.to(torch.float32)).mean(dim=-2)
        
        # pass through fully connected layers w/ activation functions
        cls_output = self.fc_cls(self.dropout(cls_output)).relu()
        e1_avg = self.fc_e(self.dropout(e1_avg)).relu()
        e2_avg = self.fc_e(self.dropout(e2_avg)).relu()
        
        # concatenate sentence level, e1, and e2
        rel_rep = torch.cat([cls_output, e1_avg, e2_avg], dim=-1)
        
        # attention across sentences in bag
        if not evaluate:
            relations = batch_data["target"]
            # query matrix for attention is the output fc weights
            attn_matrix = self.fc_class.weight[relations, :].squeeze(1).T 
            
            # compute dot product similarity between sentences and relation vectors
            sims = []
            for i in range(batch_size):
                sims.append(torch.matmul(rel_rep[i, :, :], attn_matrix[:, i]))
            sims = torch.stack(sims)
            
            # normalize sims to probabilities with sentence mask 
            sims[~(sentence_mask.to(torch.bool))] = float("-inf")
            sentence_weights = F.softmax(sims, dim=-1).unsqueeze(-1)
            # get bag sentence representation using weighted sum from attention weights
            bag_rep = torch.sum(rel_rep * sentence_weights, dim=1)
            
            # fully connected + softmax for classification
            output = self.fc_class(bag_rep)
            output = self.softmax(output)
            
        # don't know which relation to determine attention on when testing
        else:
            attn_matrix = self.fc_class.weight.T
            output = []
            for i in range(batch_size):
                bag = rel_rep[i, :] # sentences in bag x Bert representation length
                attn_scores = torch.matmul(rel_rep[i, :], attn_matrix) # sentences in bag x num_classes
                attn_scores[~(sentence_mask[i, :].to(torch.bool))] = float("-inf") # sentences in bag x num_classes
                attn_weights = F.softmax(attn_scores, dim=0) # sentences in bag x num_classes
                
                bag_rep = torch.matmul(attn_weights.T, bag)
                bag_output = self.fc_class(bag_rep)
                output.append(self.softmax(bag_output).diag())
            output = torch.stack(output)
        
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
