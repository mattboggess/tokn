from __future__ import print_function, division
import os
import torch
import json
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from relation_constants import ALL_RELATIONS

class RelationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dir, relations=ALL_RELATIONS):
        """
        Args:
            json_file (string): Path to the json file with relations.
        """
        all_relations = json.load(os.path.join(data_dir, "relations_db.json"))
        unused_relations = [relation for relation in relations_dict.keys() if relation not in relations]
        for unused_relation in unused_relations:
            all_relations.pop(unused_relation)
        self.relation_df = None 
        for relation in all_relations:
            df = pd.DataFrame.from_dict(all_relations[relation], orient='index')
            if self.relation_df is None:
                self.relation_df = df
            else:
                self.relation_df = pd.concat([self.relation_df, df])
        self.data_dir = data_dir

        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab2id = {v: id for v, id in self.vocab2id.items()}


    def __len__(self):
        return self.relation_df.size 

    def __getitem__(self, idx):
        sample = self.relation_df.iloc[idx]
        y_label = sample['relation']
        bag_of_words = [preprocess(sentence) for sentence in sample['sentences']]
        return bag_of_words, y_label 


    def preprocess(sentence_tokenized):
        sequence = [self.vocab2id(token) for token in sentence_tokenized]
        sequence = torch.Tensor(sequence)
        return sequence
