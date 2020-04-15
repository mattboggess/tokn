from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import numpy as np
import pandas as pd 
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer


class TermNERDataset(Dataset):
    """Pytorch Dataset for term named entity recognition."""

    def __init__(self, data_dir, split='train', embedding_type='Bert', max_sent_length=10,
                 tags=['O', 'S', 'B', 'I', 'E'], balance_loss=True):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the input data is stored 
        split: str 
            The data split to load. Must match the name of a data file in the data_dir 
        embedding_type: str, ['Bert'] 
            Type of embedding to use for the data loader. 
        max_sent_length: int 
            Maximum number of tokens for each sentence. Longer sentences will be truncated. 
            Shorter sentences will be padded.
        tags: list of str
            List of NER tags to be used for classification. Default is BIOES tags:
              - 'B': Beggining of term phrase
              - 'I': Interior of term phrase
              - 'O': Not a term
              - 'E': End of term phrase
              - 'S': Singleton term
        balance_loss: bool
            Whether to provide weights to the loss function to adjust for class imbalance.
        """
        data = json.load(open(os.path.join(data_dir, f"term_extraction_{split}.json")))
        self.term_counts = data['terms']
                                  
        df = {'sentence': [], 'tag': [], 'source': []}
        tag_classes = []
        for sentence, tag, source in zip(data['sentences'], data['tags'], data['sources']):
            df['sentence'].append(sentence)
            df['tag'].append(tag)
            df['source'].append(source)
            tag_classes += tag.split(' ')
        self.term_df = pd.DataFrame(df)
        
        # compute class weights to handle class imbalance
        tags = [t for t in tags if t in tag_classes]
        if balance_loss:
            self.class_weights = torch.Tensor(compute_class_weight('balanced', tags, tag_classes))
        else:
            self.class_weights = torch.Tensor([1] * len(tags))
                
        self.max_sent_length = max_sent_length
        self.embedding_type = embedding_type
        self.tags = tags
        
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased', cls_token='<cls>', pad_token='<pad>', sep_token='<sep>')

    def __len__(self):
        return self.term_df.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a single data instance that includes the following:
          - sentence: BERT-compatible tokenized sentence (input data)
          - tags: list of BIOES tags for sentence (data label)
          - pad_mask: mask denoting which tokens in sentence are padding tokens
          - bert_mask: mask denoting which tokens correspond to valid, originally tagged tokens
          - source: textbook/section source of the sentence
          - sentence_text: Original text of the sentence
        """
        sample = self.term_df.iloc[idx, :]
        
        source = sample['source'] 
        sentence, tags, pad_mask, bert_mask = self.preprocess(sample['sentence'], sample['tag'])
        sentence = torch.Tensor(sentence).to(torch.int64)
        tags = torch.Tensor(tags).to(torch.int64)
        pad_mask = torch.Tensor(pad_mask).to(torch.int64)
        bert_mask = torch.Tensor(bert_mask).to(torch.int64)
        sentence_text = sample['sentence'].split()
        
        return (sentence, tags, pad_mask, bert_mask, source, sentence_text)

    def preprocess(self, sentence, tags):
        """
        Converts input to Bert-compatible tokenized sentence that are padded/truncated to a
        pre-specified length. Additionally computes masks excluding padding and 
        
        Parameters
        ----------
        sentence: str 
            Input tokenized sentence with tokens separated by spaces 
        tags: str 
            Input list of tags for each token in the tokenized sentence
            The data split to load. Must match the name of a data file in the data_dir 
            
        Returns
        -------
        4-tuple of lists
          sequence: list of converted BERT token ids for the sentence to be fed into BERT model
          tags: list of converted tag ids to act as data labels for the BERT model
          pad_mask: mask that is 1 if token is not a padding token and 0 otherwise
          bert_mask: mask that is 1 if token corresponds to an original valid token that had a 
                     BIOES tag and 0 if it as an additional added BERT token
        """
        
        # tokenize sentence 
        tags = tags.split()
        sentence_tokenized = sentence.split()
        bert_mask = []
        if self.embedding_type == 'Bert':
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
                tmp_tags += ['O'] * (len(bert_token) - 1)
                
            sentence_tokenized = tmp_tokenized
            tags = tmp_tags
        
        # truncate long sentences 
        if len(sentence_tokenized) > self.max_sent_length - 2:
            sentence_tokenized = sentence_tokenized[:self.max_sent_length - 2]
            bert_mask = bert_mask[:self.max_sent_length - 2]
            tags = tags[:self.max_sent_length - 2]
            
        # add BERT special cls and end of sentence sep tokens
        sentence_tokenized = ['<cls>'] + sentence_tokenized + ['<sep>']
        bert_mask = [0] + bert_mask + [0] 
        tags = ['O'] + tags + ['O'] 
            
        # pad sentences & tags
        sentence_tokenized, pad_mask = self.pad_sentence(sentence_tokenized)
        tags, _ = self.pad_sentence(tags)
        tags = ['O' if tag == '<pad>' else tag for tag in tags]
        bert_mask, _ = self.pad_sentence(bert_mask)
        bert_mask = [0 if tag == '<pad' else tag for tag in bert_mask]
        
        # convert to embedding ids
        sequence = self.tokenizer.convert_tokens_to_ids(sentence_tokenized)
        
        # convert labels to numerical values
        tags = [self.tags.index(tag) for tag in tags]
            
        return sequence, tags, pad_mask, bert_mask
    
    def pad_sentence(self, sentence_tokenized, pad_token='<pad>'):
        """Pad end of tokenized sentence with special padding token to reach pre-specified length"""
        
        for _ in range(self.max_sent_length - len(sentence_tokenized)):
            sentence_tokenized.append('<pad>')
        pad_mask = [0 if tok == '<pad>' else 1 for tok in sentence_tokenized]
        return sentence_tokenized, pad_mask
            
class TermNERDataLoader(DataLoader):
    """
    Pytorch data loader for term extraction Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=0,
                 split='train', tags=['O', 'S', 'B', 'I', 'E'], embedding_type='Bert', 
                 max_sent_length=10):
        """
        Parameters
        ----------
        data_dir: str 
            Path to where the term extraction input data is stored 
        batch_size: int
            Number of sentences in each batch 
        shuffle: bool
            Whether to shuffle the order of the data being loaded in
        num_workers: int 
            Number of workers to use to read in data in parallel 
        split: str
            The data split to load. Must match the name of a data file in the data_dir 
        tags: list of str
            List of NER tags to be used for classification. Default is BIOES tags:
              - 'B': Beggining of term phrase
              - 'I': Interior of term phrase
              - 'O': Not a term
              - 'E': End of term phrase
              - 'S': Singleton term
        embedding_type: str, ['Bert'] 
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
        """
        Function used to collect individual sentences into a batch. Each batch is
        a dictionary that contains stacked tensors for the input data, data labels, and data masks.
        Additionally, it includes the original text and source for each sentence in the batch.
        """
        
        batch_data = {
            'data': torch.stack([bd[0] for bd in batch_data]),
            'target': torch.stack([bd[1] for bd in batch_data]).squeeze(0),
            'pad_mask': torch.stack([bd[2] for bd in batch_data]),
            'bert_mask': torch.stack([bd[3] for bd in batch_data]),
            'sources': [bd[4] for bd in batch_data],
            'sentences': [bd[5] for bd in batch_data]
        }
        
        return batch_data 
