from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import numpy as np
import pandas as pd 
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer


class TermNERDataset(Dataset):
    """Dataset for term named entity recognition."""

    def __init__(self, data_dir, split="train", embedding_type="Bert", max_sent_length=10,
                 tags=["O", "S", "B", "I", "E"]):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        split: str, ['train', 'validation', life_test', 'psych_test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        relations: list of str 
            List of relations to include to classify between 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. 
            Shorter sentences will be padded.
        tags: list of str
            List of NER tags to be used for classification:
              - 'B': Beggining of term phrase
              - 'I': Interior of term phrase
              - 'E': End of term phrase
              - 'S': Singleton term
              - 'O': Not a term
        """
        data = json.load(open(os.path.join(data_dir, f"term_extraction_{split}.json")))
        self.term_counts = data["terms"]
                                  
        df = {"sentence": [], "tag": [], "textbook": []}
        tag_classes = []
        for sentence, tag, textbook in zip(data["sentences"], data["tags"], data["textbook"]):
            df["sentence"].append(sentence)
            df["tag"].append(tag)
            df["textbook"].append(textbook)
            tag_classes += tag.split(" ")
        self.term_df = pd.DataFrame(df)
        
        # compute class weights to handle class imbalance
        tags = [t for t in tags if t in tag_classes]
        self.class_weights = torch.Tensor(compute_class_weight("balanced", tags, tag_classes))
                
        self.max_sent_length = max_sent_length
        self.embedding_type = embedding_type
        self.tags = tags
        
        if self.embedding_type == "custom":
            self.vocab2id = json.load(open(os.path.join(data_dir, 'word2id.json')))
        elif self.embedding_type == "Bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased", cls_token="<cls>", pad_token="<pad>", sep_token="<end>")

    def __len__(self):
        return self.term_df.shape[0]

    def __getitem__(self, idx):
        sample = self.term_df.iloc[idx, :]
        
        textbook = sample["textbook"] 
        sentence, tags, pad_mask, bert_mask = self.preprocess(sample["sentence"], sample["tag"])
        sentence = torch.Tensor(sentence).to(torch.int64)
        tags = torch.Tensor(tags).to(torch.int64)
        pad_mask = torch.Tensor(pad_mask).to(torch.int64)
        bert_mask = torch.Tensor(bert_mask).to(torch.int64)
        sentence_text = sample["sentence"].split()
        
        return (sentence, tags, pad_mask, bert_mask, textbook, sentence_text)

    def preprocess(self, sentence, tags):
        
        # tokenize sentence 
        tags = tags.split()
        sentence_tokenized = sentence.split()
        bert_mask = []
        if self.embedding_type == "Bert":
            tmp_tokenized = []
            tmp_tags = []
            for token, tag in zip(sentence_tokenized, tags):
                
                # use Bert tokenizer on each token
                bert_token = self.tokenizer.tokenize(token)
                tmp_tokenized += bert_token
                
                # add mask for mapping to original tokens
                bert_mask += [1] 
                bert_mask += [0] * (len(bert_token) - 1)
                
                # pad tags with empty tags to make all same length
                tmp_tags += [tag]
                tmp_tags += ["O"] * (len(bert_token) - 1)
                
            sentence_tokenized = tmp_tokenized
            tags = tmp_tags
        
        # truncate long sentences 
        if len(sentence_tokenized) > self.max_sent_length - 2:
            sentence_tokenized = sentence_tokenized[:self.max_sent_length - 2]
            bert_mask = bert_mask[:self.max_sent_length - 2]
            tags = tags[:self.max_sent_length - 2]
            
        # add beginning and end of sentence tokens
        if self.embedding_type == "Bert":
            sentence_tokenized = ["<cls>"] + sentence_tokenized + ["<end>"]
            bert_mask = [0] + bert_mask + [0] 
            tags = ["O"] + tags + ["O"] 
        else:
            sentence_tokenized += ["<end>"]
            
        # pad sentences & tags
        sentence_tokenized, pad_mask = self.pad_sentence(sentence_tokenized)
        tags, _ = self.pad_sentence(tags)
        tags = ["O" if tag == "<pad>" else tag for tag in tags]
        bert_mask, _ = self.pad_sentence(bert_mask)
        bert_mask = [0 if tag == "<pad>" else tag for tag in bert_mask]
        
        # convert to embedding ids
        if self.embedding_type == "Bert":
            sequence = self.tokenizer.convert_tokens_to_ids(sentence_tokenized)
        elif self.embedding_type == "custom":
            sequence = [self.vocab2id[token.lower()] 
                        if token.lower() in self.vocab2id else self.vocab2id['<unk>'] 
                        for token in sentence_tokenized]
        
        # convert labels to numerical values
        tags = [self.tags.index(tag) for tag in tags]
            
        return sequence, tags, pad_mask, bert_mask
    
    def pad_sentence(self, sentence_tokenized, pad_token="<pad>"):
        """pad end of sentences to match same length"""
        for _ in range(self.max_sent_length - len(sentence_tokenized)):
            sentence_tokenized.append("<pad>")
        pad_mask = [0 if tok == "<pad>" else 1 for tok in sentence_tokenized]
        return sentence_tokenized, pad_mask
            
class TermNERDataLoader(DataLoader):
    """
    Data loader for term extraction 
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=0,
                 split="train", tags=["O", "S", "B", "I", "E"], embedding_type="Bert", 
                 max_sent_length=10):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the relations data is stored 
        batch_size: int
            Number of word-pairs in each batch 
        sampler: torch.utils.data.Sampler
            Pytorch sampler object used to sample the data
        relations: list of str 
            List of relations to include to classify between 
        shuffle: bool
            Whether to shuffle the order of the data being loaded in
        num_workers: int 
            Number of workers to use to read in data in parallel 
        split: str, ['train', 'validation', life_test', 'psych_test', 'debug'] 
            The data split to load. debug is a small debugging dataset 
        embedding_type: str, ['Bert', 'custom'] 
            Type of embedding to use for the data. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. 
            Shorter sentences will be padded.
        """
        self.data_dir = data_dir
        self.tags = tags
        self.dataset = TermNERDataset(self.data_dir, split=split, embedding_type=embedding_type,
                                      tags=tags, max_sent_length=max_sent_length)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=self.collate_fn)
        
    def collate_fn(self, batch_data):
        
        batch_data = {
            "data": torch.stack([bd[0] for bd in batch_data]),
            "target": torch.stack([bd[1] for bd in batch_data]).squeeze(0),
            "pad_mask": torch.stack([bd[2] for bd in batch_data]),
            "bert_mask": torch.stack([bd[3] for bd in batch_data]),
            "textbooks": [bd[4] for bd in batch_data],
            "sentences": [bd[5] for bd in batch_data]
        }
        
        return batch_data 
