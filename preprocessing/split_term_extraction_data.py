# Splits term-tagged sentences from various textbook sources into training, validation, and test
# sets for training and evaluating term extraction models.

# Author: Matthew Boggess
# Version: 4/13/20

# Data Source: Output of tag_sentences.py.

# Description: 
#   - User specifies a splits parameter that maps different data splits to sets of input
#     textbooks/sections sources. All of the sentences, tags, and terms for each split are 
#     aggregated into data split json structures to be input to the models. 
#   - Additionally, all sentences without any tagged terms are removed and sentences from the train 
#     split are removed if they have any tagged terms that are in validation or test splits.
#   - A summary dataframe containing sentence and term counts for each of the data sources 
#     is also compiled and saved.

#===================================================================================

# Libraries 

import spacy
from data_processing_utils import tag_terms, read_spacy_docs
from collections import Counter
from tqdm import tqdm
import os
import json
import pandas as pd

#===================================================================================

# Parameters

## Filepaths

input_data_dir = "../data/preprocessed/tagged_sentences"
output_data_dir = "../data/term_extraction"

## Other parameters 

# mapping from data splits to data sources 
splits = {
    'train': [
        ('Life_Biology', 'all', 'all'),
        ('Biology_2e', 'all', 'all')
        #('Anatomy_and_Physiology', 'all', 'all'), 
        #('Astronomy', 'all', 'all'), 
        #('Chemistry_2e', 'all', 'all'), 
        #('Microbiology', 'all', 'all'), 
        #('University_Physics_Volume_1', 'all', 'all'),
        #('University_Physics_Volume_2', 'all', 'all'),
        #('University_Physics_Volume_3', 'all', 'all')
    ],
    'dev': [('dev', 'all', 'all')],
    'test': [('test', 'all', 'all')]
}

# flag on whether sentences without any key terms tagged should be excluded
ignore_empty_sentences = True

# 1/3 length of small debugging set to create
debug_length = 20

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    splits_data = {}
    for split, sources in splits.items():
        
        split_data = []
        for textbook, chapters, sections in sources:
            data = pd.read_pickle(f"{input_data_dir}/{textbook}_tagged_sentences.pkl")
            
            # filter to chapter and section selections
            if chapters == 'all':
                chapters = sorted(list(data.chapter.unique()))
            if sections == 'all':
                sections = sorted(list(data.section.unique()))
            data = data[(data.chapter.isin(chapters) & (data.section.isin(sections)))]
            
            if ignore_empty_sentences:
                data = data[data.terms.apply(lambda x: len(x) > 0)]
                
            split_data.append(data)
        splits_data[split] = pd.concat(split_data) 
    
    # remove overlap between train and dev/test sets
    eval_set = set()
    for terms in splits_data['dev'].terms:
        eval_set = eval_set | terms
    for terms in splits_data['test'].terms:
        eval_set = eval_set | terms
    splits_data['train'] = splits_data['train'][splits_data['train'].terms.apply( \
        lambda x: len(x & eval_set) == 0)]
    
    # create small debug set
    b_samples = splits_data['train'][splits_data['train'].tags.apply(lambda x: 'B' in x)].sample(debug_length)
    i_samples = splits_data['train'][splits_data['train'].tags.apply(lambda x: 'I' in x)].sample(debug_length)
    s_samples = splits_data['train'][splits_data['train'].tags.apply(lambda x: 'S' in x)].sample(debug_length)
    splits_data['debug'] = pd.concat([b_samples, i_samples, s_samples]).copy()
        
    # write out data splits
    for split in splits_data:
        splits_data[split].to_pickle(f"{output_data_dir}/term_extraction_{split}.pkl")
