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

    def __init__(self, data_dir, split, label_type, balance_loss=False, max_sent_length=256):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'dev', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset. 
        label_type: str, 'hard_label' or 'soft_label'
            Type of label to use. Hard means each label is a particular class, soft means each label is a probability dist across classes 
        balance_loss: bool
            Indicator whether the loss function should be weighted to account for class imbalance
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        # read in data as pandas dataframe
        input_file = f"{data_dir}/{split}.pkl"
        with open(input_file, 'rb') as fid:
            self.data = pickle.load(fid)
                                  
        # handle labels 
        self.label_column = label_type 
        
        # get ordered list of unique label types
        tmp = sorted(list(set([(x, y) for (x, y) in zip(self.data.hard_label, self.data.hard_label_class)])), key=lambda x: x[0])
        self.relation_classes = [x[1] for x in tmp]
        self.relations = [x[0] for x in tmp]
        self.num_classes = len(self.relations)
        
        # compute class weights to balance loss
        if balance_loss:
            relation_list = list(self.data.hard_label)
            self.class_weights = torch.Tensor(
                compute_class_weight('balanced', self.relations, relation_list))
        else:
            self.class_weights = torch.Tensor([1.0] * len(self.relations))
                
        self.max_sent_length = max_sent_length
        
        # set up BERT representations (add entity markers)
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

        # target goes to loss, label goes to metrics
        target = sample[self.label_column]
        if self.label_column == 'hard_label':
            target = [target]
            label = target
        else:
            label = [sample['hard_label']]
     
        input_text = sample.sentence
        term_pair = sample.term_pair
        relation = self.relation_classes[self.relations.index(label[0])]
        
        # add term start and end tokens 
        tokens = [tok for tok in sample.tokens]
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
        
        # create term start location masks (if a term is cut off by max sentence truncation
        # simply 0 out the attention mask so that example is ignored)
        term1_id = self.tokenizer.convert_tokens_to_ids(self.term1_start_token)
        term2_id = self.tokenizer.convert_tokens_to_ids(self.term2_start_token)
        if term1_id in bert_input_ids:
            term1_mask = [1 if tok == term1_id else 0 for tok in bert_input_ids] 
        else:
            print('Missing Term 1')
            term1_mask = [0] * len(bert_input_ids)
            bert_attention_mask *= 0
        if term2_id in bert_input_ids:
            term2_mask = [1 if tok == term2_id else 0 for tok in bert_input_ids] 
        else:
            print('Missing Term 2')
            term2_mask = [0] * len(bert_input_ids)
            bert_attention_mask *= 0
        
        output = {
            'input_ids': torch.LongTensor(bert_input_ids),
            'attention_mask': torch.LongTensor(bert_attention_mask),
            'bert_representation': bert_representation,
            'text': input_text,
            'term_pair': term_pair,
            'relation': relation, 
            'term1_mask': torch.FloatTensor(term1_mask),
            'term2_mask': torch.FloatTensor(term2_mask),
            'label': torch.LongTensor(label),
            'target': torch.FloatTensor(target)
        }
        
        return output
            
class RelationDataLoader(DataLoader):
    """
    Data loader for Relations Dataset for weakly supervised relation extraction. 
    """
    def __init__(self, data_dir, split, label_type, batch_size, shuffle=True,  
                 num_workers=1,  max_sent_length=256, balance_loss=False):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'dev', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset. 
        label_type: str, 'hard_label' or 'soft_label'
            Type of label to use. Hard means each label is a particular class, soft means each label is a probability dist across classes 
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
            label_type,
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
            
        tensor_fields = ['input_ids', 'attention_mask', 'target', 'label', 'term1_mask', 'term2_mask'] 
        for tf in tensor_fields:
            output[tf] = torch.stack(output[tf])
            
        return output 
