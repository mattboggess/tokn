from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import numpy as np
import pandas as pd 
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer

RELATION_MAPPING = {
    "taxonomy": ["subclass-of"],
    "meronym": ["has-part", "has-region", "element", "possesses", "material"],
    "spatial": ["is-at", "is-between", "is-inside", "is-outside", "abuts"],
    "event_structure": ["first-subevent", "subevent", "next-event"],
    "participant": ["agent", "object", "instrument", "raw-material", "result", "site"],
    "causal": ["causes", "enables", "prevents", "inhibits"]
}

class RelationDataset(Dataset):
    """Relation dataset."""

    def __init__(self, data_dir, relations, split="train", embedding_type="Bert", 
                 max_sent_length=10, predict=False):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        relations: list of str 
            List of relations to include to classify between 
        split: str, ['train', 'validation', 'test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. Shorter
            sentences will be padded.
        """
        
        all_relations = json.load(open(os.path.join(data_dir, f"relations_{split}.json")))
        
        # all relations are dummy coded as no-relation if predicting on new data
        if predict:
            all_relations = {k: v for k,v in all_relations.items() if k in ["no-relation"]}
        # read in relations consolidating those in same family if family is passed 
        else:
            tmp = {}
            for relation in relations:
                if relation in RELATION_MAPPING:
                    tmp[relation] = []
                    for r in RELATION_MAPPING[relation]:
                        if r in all_relations:
                            tmp[relation] += all_relations[r]
                else:
                    tmp[relation] = all_relations[relation]
            all_relations = tmp
                                  
        # add no-relation and convert into dataframe
        self.relations = ["no-relation"] + relations
        self.relation_df = None 
        for relation in all_relations:
            for i in range(len(all_relations[relation])):
                df = pd.DataFrame.from_dict(all_relations[relation][i], orient='index')
                df.loc[df.relation != "no-relation", "relation"] = relation
                if self.relation_df is None:
                    self.relation_df = df
                else:
                    self.relation_df = pd.concat([self.relation_df, df], sort=False)
        
        # compute weights to adjust for class imbalance
        if not predict:
            self.class_weights = torch.Tensor(compute_class_weight("balanced", self.relations, 
                                                                   self.relation_df.relation))
        else:
            self.class_weights = torch.Tensor([1.0] * len(self.relations))
                
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
    def __init__(self, data_dir, batch_size, relations, shuffle=True, max_sentences=16, 
                 num_workers=1, split="train", embedding_type="custom", max_sent_length=10,
                 predict=False):
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
        self.max_sentences = max_sentences
        self.dataset = RelationDataset(self.data_dir, split=split, embedding_type=embedding_type,
                                       relations=relations, max_sent_length=max_sent_length,
                                       predict=predict)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=self.relation_collate_fn)
        
    def relation_collate_fn(self, batch_data):
        
        #max_input_sentences = max([bd[0].shape[0] for bd in batch_data]) 
        #max_input_sentences = int(np.ceil(max_input_sentences / self.max_sentences) * self.max_sentences)
        
        output_fields = ["data", "target", "pad_mask", "e1_mask", "e2_mask", "sentence_mask"]
        output = {k: [] for k in output_fields}
        output["word_pair"] = []
        for bd in batch_data:
            if len(bd[0].shape) < 2:
                continue
            output["data"].append(self.pad_sentences(bd[0]))
            output["target"].append(bd[1])
            output["pad_mask"].append(self.pad_sentences(bd[3]))
            output["e1_mask"].append(self.pad_sentences(bd[4]))
            output["e2_mask"].append(self.pad_sentences(bd[5]))
            output["word_pair"].append(bd[2])
            
            # add sentence mask
            num_pad = self.max_sentences - bd[0].shape[0]
            sent_mask = torch.Tensor([1] * min(self.max_sentences, bd[0].shape[0]) + [0] * num_pad)
            output["sentence_mask"].append(sent_mask)
            
        if not len(output["data"]):
            return None
        batch_output = {k: torch.stack(output[k]) for k in output_fields}
        batch_output["word_pair"] = output["word_pair"]
        
        return batch_output 
    
    def pad_sentences(self, data):
        if self.max_sentences >= data.shape[0]:
            num_pad = self.max_sentences - data.shape[0]
            padding = torch.Tensor(np.zeros((num_pad, data.shape[1]))).to(torch.int64)
            data = torch.cat((data, padding), 0)
        else:
            data = data[:self.max_sentences, :]
        return data
