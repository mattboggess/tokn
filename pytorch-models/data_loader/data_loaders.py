from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import numpy as np
import pandas as pd 

ALL_RELATIONS = ['subclass-of', 'has-part', 'possesses', 'has-region', 'is-inside', 'is-at', 'element', 'abuts', 'is-outside']
TAXONOMY_RELATIONS = ['subclass-of']
STRUCTURE_RELATIONS = ['has-part', 'possesses', 'has-region', 'is-inside', 'is-at', 'element', 'abuts', 'is-outside']


class RelationDataset(Dataset):
    """Relation dataset."""

    def __init__(self, data_dir, split="train", relations=ALL_RELATIONS, embedding_type="Bert",
                 max_sent_length=10):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        relations: list of str 
            List of relations to include to classify between 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        all_relations = json.load(open(os.path.join(data_dir, f"relations_{split}.json")))
        all_relations = {k: v for k,v in all_relations.items() if k in relations}
                                  
        self.relations = ["no-relation"] + relations
        self.relation_df = None 
        for relation in all_relations:
            for i in range(len(all_relations[relation])):
                df = pd.DataFrame.from_dict(all_relations[relation][i], orient='index')
                if self.relation_df is None:
                    self.relation_df = df
                else:
                    self.relation_df = pd.concat([self.relation_df, df], sort=False)
                
        self.max_sent_length = max_sent_length
        self.embedding_type = embedding_type
        if self.embedding_type == "custom":
            self.vocab2id = json.load(open(os.path.join(data_dir, 'word2id.json')))

    def __len__(self):
        return self.relation_df.shape[0]

    def __getitem__(self, idx):
        sample = self.relation_df.iloc[idx, :]
        y_label = self.relations.index(sample["relation"]) 
        y_label = torch.Tensor([y_label]).to(torch.int64)
        word_pair = self.relation_df.index[idx]
        
        bag_of_words = [self.preprocess(sentence) for sentence in sample['sentences']]
        bag_of_words = torch.Tensor(bag_of_words).to(torch.int64)
        return (bag_of_words, y_label, word_pair)

    def preprocess(self, sentence):
        sentence_tokenized = sentence.split(" ")
        
        # add Bert sentence start token
        if self.embedding_type == "Bert":
            sentence_tokenized = ["[CLS]"] + sentence_tokenized
        
        # truncate or pad end of sentences to make same length
        # TODO: Add padding mask so we actually ignore padded values when computing
        if len(sentence_tokenized) > self.max_sent_length:
            sentence_tokenized = sentence_tokenized[:self.max_sent_length]
        else:
            for _ in range(self.max_sent_length - len(sentence_tokenized)):
                sentence_tokenized.append("<pad>")
        
        if self.embedding_type == "custom":
            sequence = self._preprocess_custom(sentence_tokenized)
            
        return sequence
    
    def _preprocess_custom(self, sentence_tokenized):
        sequence = [self.vocab2id[token.lower()] 
                    if token in self.vocab2id else self.vocab2id['<unk>'] 
                    for token in sentence_tokenized]
        return sequence
            
class RelationDataLoader(BaseDataLoader):
    """
    Data loader for biology relations
    """
    def __init__(self, data_dir, batch_size, relations, shuffle=True, validation_split=0.0, 
                 num_workers=1, split="train", embedding_type="custom", max_sent_length=10):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        batch_size: int
            Number of word-pairs in each batch (TODO: Support > 1 batch size)
        relations: list of str 
            List of relations to include to classify between 
        shuffle: bool
            Whether to shuffle the order of the data being loaded in
        validation_split: float, [0, 1] 
            Fraction of data to hold out for validation 
        num_workers: int 
            Number of workers to use to read in data in parallel 
        split: str, ['train', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        self.data_dir = data_dir
        self.dataset = RelationDataset(self.data_dir, split=split, embedding_type=embedding_type,
                                       relations=relations, max_sent_length=max_sent_length)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         collate_fn = self.relation_collate_fn)
        
    def relation_collate_fn(self, batch_data):
        # TODO: How do we handle batches since each bag has a different number of sentences?
        #       Probably need to add "padding" with masks just like we do with sentences
        
        input_data = torch.stack([bd[0] for bd in batch_data])
        target = torch.stack([bd[1] for bd in batch_data]).squeeze(0)
        word_pairs = [bd[2] for bd in batch_data]
        
        return (input_data, target, word_pairs)
