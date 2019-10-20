from __future__ import print_function, division
import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from relation_constants import ALL_RELATIONS

class RelationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, json_file, relations=ALL_RELATIONS, transform=None):
        """
        Args:
            json_file (string): Path to the json file with relations.
        """
        all_relations = json.load(json_file)
        unused_relations = [relation for relation in relations_dict.keys() if relation not in relations]
        for unused_relation in unused_relations:
            all_relations.pop(unused_relation)
        self.relations = all_relations.values()
        self.transform = transform

    def __len__(self):
        return sum([len(self.relations) for relation in self.relations])

    def __getitem__(self, idx):
        pass 