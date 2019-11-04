from __future__ import print_function, division
import os
import torch
import json
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from relation_constants import ALL_RELATIONS

class RelationDataset(Dataset):
    """Relation dataset."""

    def __init__(self, data_dir, split="train", relations=ALL_RELATIONS, embedding_type="Bert"):
        """
        Args:
            data_dir (string): Path to the data directory with relations_db json file and words.txt 
                               file for vocab2id generation
            relations (string): list of relations 
        """
        all_relations = json.load(open(os.path.join(data_dir, f"relations_{split}.json")))
        unused_relations = [relation for relation in all_relations.keys() 
                            if relation not in relations]
        all_relations = {k: v for k,v in all_relations.items() if k in relations}
                                  
        self.relations = relations
        self.relation_df = None 
        for relation in all_relations:
            for i in range(len(all_relations[relation])):
                df = pd.DataFrame.from_dict(all_relations[relation][i], orient='index')
                if self.relation_df is None:
                    self.relation_df = df
                else:
                    self.relation_df = pd.concat([self.relation_df, df], sort=False)
                
        self.embedding_type = embedding_type
        if self.embedding_type == "custom":
            self.vocab2id = json.load(open(os.path.join(data_dir, 'word2id.json')))

    def __len__(self):
        return self.relation_df.shape[0]

    def __getitem__(self, idx):
        sample = self.relation_df.iloc[idx]
        y_label = [1 if sample['relation'] == relation else 0 for relation in self.relations]
        word_pair = self.relation_df.index
        
        bag_of_words = [self.preprocess(sentence) for sentence in sample['sentences']]
        return (torch.Tensor(bag_of_words), y_label, word_pair)

    def preprocess(self, sentence, text_length=50):
        sentence_tokenized = sentence.split(" ")
        
        # pad end of sentences to make same length
        if len(sentence_tokenized) > text_length:
            sentence_tokenized = sentence_tokenized[:text_length]
        else:
            for _ in range(text_length - len(sentence_tokenized)):
                sentence_tokenized.append("<pad>")
        
        if self.embedding_type == "custom":
            sequence = self._preprocess_custom(sentence_tokenized)
            
        return sequence
    
    def _preprocess_custom(self, sentence_tokenized):
        sequence = [self.vocab2id[token.lower()] 
                    if token in self.vocab2id else self.vocab2id['<unk>'] 
                    for token in sentence_tokenized]
        return sequence
            
