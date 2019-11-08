from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import numpy as np
import pandas as pd 
from transformers import BertTokenizer

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
        split: str, ['train', 'validation', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        relations: list of str 
            List of relations to include to classify between 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        
        # filter to relations classifying between
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
        elif self.embedding_type == "Bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased", cls_token="<cls>", pad_token="<pad>", sep_token="<end>",
                additional_special_tokens=["<e1>", "</e1>", "<e2>", "</e2>"])

    def __len__(self):
        return self.relation_df.shape[0]

    def __getitem__(self, idx):
        sample = self.relation_df.iloc[idx, :]
        y_label = self.relations.index(sample["relation"]) 
        y_label = torch.Tensor([y_label]).to(torch.int64)
        word_pair = self.relation_df.index[idx]
        
        bag_of_words = []
        pad_mask = []
        e1_mask = []
        e2_mask = []
        for sentence in sample['sentences']:
            preprocess_result = self.preprocess(sentence)
            if preprocess_result is None:
                continue 
            
            bag_of_words.append(preprocess_result[0])
            pad_mask.append(preprocess_result[1])
            e1_mask.append(preprocess_result[2])
            e2_mask.append(preprocess_result[3])
                           
        bag_of_words = torch.Tensor(bag_of_words).to(torch.int64)
        pad_mask = torch.Tensor(pad_mask).to(torch.int64)
        e1_mask = torch.Tensor(e1_mask).to(torch.int64)
        e2_mask = torch.Tensor(e2_mask).to(torch.int64)
        
        return (bag_of_words, y_label, word_pair, pad_mask, e1_mask, e2_mask)

    def preprocess(self, sentence):
        
        # tokenize sentence 
        if self.embedding_type == "Bert":
            sentence_tokenized = self.tokenizer.tokenize(sentence)
        elif self.embedding_type == "custom":
            sentence_tokenized = sentence.split(" ")
        
        # truncate long sentences and add end special sentence boundary tokens 
        if len(sentence_tokenized) > self.max_sent_length - 2:
            sentence_tokenized = sentence_tokenized[:self.max_sent_length - 2]
        sentence_tokenized = ["<cls>"] + sentence_tokenized + ["<end>"]
        
        # pad sentences
        sentence_tokenized, pad_mask = self.pad_sentence(sentence_tokenized)
        
        # get word pair masks
        if "</e1>" not in sentence_tokenized or "</e2>" not in sentence_tokenized:
            return None
        e1_index = (sentence_tokenized.index("<e1>"), sentence_tokenized.index("</e1>"))
        e1_mask = [1 if (i > e1_index[0] and i < e1_index[1]) else 0 
                   for i in range(len(sentence_tokenized))]
        e2_index = (sentence_tokenized.index("<e2>"), sentence_tokenized.index("</e2>"))
        e2_mask = [1 if (i > e2_index[0] and i < e2_index[1]) else 0 
                   for i in range(len(sentence_tokenized))]
        
        # convert to embedding ids
        if self.embedding_type == "Bert":
            sequence = self.tokenizer.convert_tokens_to_ids(sentence_tokenized)
        elif self.embedding_type == "custom":
            sequence = [self.vocab2id[token.lower()] 
                        if token in self.vocab2id else self.vocab2id['<unk>'] 
                        for token in sentence_tokenized]
            
        return sequence, pad_mask, e1_mask, e2_mask
    
    def pad_sentence(self, sentence_tokenized, pad_token="<pad>"):
        """pad end of sentences to match same length"""
        for _ in range(self.max_sent_length - len(sentence_tokenized)):
            sentence_tokenized.append("<pad>")
        pad_mask = [0 if tok == "<pad>" else 1 for tok in sentence_tokenized]
        return sentence_tokenized, pad_mask
            
class RelationDataLoader(DataLoader):
    """
    Data loader for biology relations
    """
    def __init__(self, data_dir, batch_size, relations, shuffle=True, 
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
        num_workers: int 
            Number of workers to use to read in data in parallel 
        split: str, ['train', 'validation', test', 'debug'] 
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
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=self.relation_collate_fn)
        
    def relation_collate_fn(self, batch_data):
        # TODO: How do we handle batches since each bag has a different number of sentences?
        #       Probably need to add "padding" with masks just like we do with sentences
        
        output = {}
        output["data"] = torch.stack([bd[0] for bd in batch_data])
        output["target"] = torch.stack([bd[1] for bd in batch_data]).squeeze(0)
        output["word_pairs"] = [bd[2] for bd in batch_data]
        output["pad_mask"] = torch.stack([bd[3] for bd in batch_data])
        output["e1_mask"] = torch.stack([bd[4] for bd in batch_data])
        output["e2_mask"] = torch.stack([bd[5] for bd in batch_data])
        
        return output 
