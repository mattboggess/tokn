from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch
import numpy as np
import pandas as pd 
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer

class RelationDataset(Dataset):
    """PyTorch Dataset for the weakly supervised relation extraction task. 
       Each data point is a sentence with two terms tagged and a relation type holding 
       between them."""

    def __init__(self, data_dir, split, label_column, balance_loss=False, max_sent_length=256):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'dev', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset. 
        label_column: str
            Column to use for y-labels
        balance_loss: bool
            Indicator whether the loss function should be weighted to account for class imbalance
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        input_file = f"{data_dir}/{split}.pkl"
        with open(input_file, 'rb') as fid:
            self.data = pickle.load(fid)
                                  
        self.relations = sorted(np.unique(self.data.majority_vote))
        self.num_classes = len(self.relations)
        relation_list = list(self.data.majority_vote)
        
        if balance_loss:
            self.class_weights = torch.Tensor(
                compute_class_weight('balanced', self.relations, relation_list))
        else:
            self.class_weights = torch.Tensor([1.0] * len(self.relations))
                
        self.max_sent_length = max_sent_length
        self.label_column = label_column
        
        # set up BERT representations
        self.term1_start_token = '[E1start]'
        self.term1_end_token = '[E1end]'
        self.term2_start_token = '[E2start]'
        self.term2_end_token = '[E2end]'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', padding_side='right')
        self.tokenizer.add_tokens(
            [self.term1_start_token, self.term1_end_token, 
             self.term2_start_token, self.term2_end_token])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the following for each data point:
          - term pair: tuple denoting the 
          - term1_location: tuple denoting the index of the start and end of term1 tokens 
          - term2_location: tuple denoting the index of the start and end of term2 tokens 
          - bert_token_ids: input text converted to Bert token ids
          - attention_mask: mask denoting which entries of the input are padding
          - text: string representation of the input text
          - label: y-label for this data point (relation type)
        """
        sample = self.data.iloc[idx, :]
        label = sample[self.label_column]
        term_pair = sample.term_pair
        if type(label) != list:
            label = [label]
        
        # add term start and end tokens 
        tokens = sample.tokens
        term1_start_idx, term1_end_idx = sample.term1_location
        term2_start_idx, term2_end_idx = sample.term2_location
        tokens[term1_start_idx] = f"{self.term1_start_token} {tokens[term1_start_idx]}"
        tokens[term1_end_idx - 1] = f"{tokens[term1_end_idx - 1]} {self.term1_end_token}"
        tokens[term2_start_idx] = f"{self.term2_start_token} {tokens[term2_start_idx]}"
        tokens[term2_end_idx - 1] = f"{tokens[term2_end_idx - 1]} {self.term2_end_token}"
        text = ' '.join(tokens)
        
        
        # convert input text to BERT token ids
        bert_ids = self.tokenizer.tokenize(text)
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_ids)
        bert_ids = self.tokenizer.prepare_for_model(
            bert_ids, 
            max_length=self.max_sent_length,
            pad_to_max_length=True
        )
        bert_representation = self.tokenizer.decode(bert_ids['input_ids'])
        bert_input_ids = bert_ids['input_ids']
        bert_attention_mask = bert_ids['attention_mask']
        
        # create term start location masks
        t1_start_idx = bert_input_ids.index(self.tokenizer.convert_tokens_to_ids(self.term1_start_token))
        term1_mask = [1 if idx == t1_start_idx else 0 for idx in range(len(bert_input_ids))] 
        t2_start_idx = bert_input_ids.index(self.tokenizer.convert_tokens_to_ids(self.term2_start_token))
        term2_mask = [1 if idx == t2_start_idx else 0 for idx in range(len(bert_input_ids))] 
        
        output = {
            'input_ids': torch.LongTensor(bert_input_ids),
            'attention_mask': torch.LongTensor(bert_attention_mask),
            'bert_representation': bert_representation,
            'text': text,
            'term_pair': term_pair,
            'term1_mask': torch.LongTensor(term1_mask),
            'term2_mask': torch.LongTensor(term2_mask),
            'label': torch.LongTensor(label)
        }
        
        return output
            
class RelationDataLoader(DataLoader):
    """
    Data loader for Relations Dataset for weakly supervised relation extraction. 
    """
    def __init__(self, data_dir, split, label_column, batch_size, shuffle=True,  
                 num_workers=1,  max_sent_length=256, balance_loss=False):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'dev', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset. 
        label_column: str
            Column to use for y-labels
        batch_size: int
            Number of word-pairs in each batch (TODO: Support > 1 batch size)
        shuffle: bool
            Whether to shuffle the order of the data being loaded in
        num_workers: int 
            Number of workers to use to read in data in parallel 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        balance_loss: bool
            Indicator whether the loss function should be weighted to account for class imbalance
        """
        self.data_dir = data_dir
        self.dataset = RelationDataset(
            self.data_dir, 
            split, 
            label_column,
            balance_loss=balance_loss,
            max_sent_length=max_sent_length
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=self.relation_collate_fn)
        
    def relation_collate_fn(self, batch_data):
        """
        Aggregrates multiple word pairs into a single batch. Also adds padded sentences at
        the bag level so that each word pair bag has the same number of sentences.
        """
        output = {k: [] for k in batch_data[0].keys()}
        for bd in batch_data:
            for k in bd.keys():
                output[k].append(bd[k])
            
        tensor_fields = ['input_ids', 'attention_mask', 'label', 'term1_mask', 'term2_mask'] 
        for tf in tensor_fields:
            output[tf] = torch.stack(output[tf])
            
        return output 
