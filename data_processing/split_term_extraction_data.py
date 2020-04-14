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

input_data_dir = "../data/term_extraction/tagged_sentences"
summary_data_dir = "../data/term_extraction/summary_data"
if not os.path.exists(summary_data_dir):
    os.makedirs(summary_data_dir)
output_data_dir = "../data/term_extraction"

## Other parameters 

# mapping from data splits to data sources 
splits = {
    'train': [
        'Anatomy_and_Physiology', 
        'Astronomy', 
        'Chemistry_2e',
        'Microbiology',
        'University_Physics_Volume_1',
        'University_Physics_Volume_2',
        'University_Physics_Volume_3',
    ],
    'validation': [
        'openstax_bio2e_section10-2',
        'openstax_bio2e_section10-4',
        'openstax_bio2e_section4-2'
    ],
    'life_kb': ['Life_Biology_kb'],
    'openstax_bio2e': ['Biology_2e'],
    'openstax_psych': ['Psychology']
}

# splits for which we want to ensure there are no terms in the train set that overlap
train_exclude = ['validation', 'openstax_bio2e', 'openstax_psych', 'life_kb']

# flag on whether sentences without any key terms tagged should be excluded
ignore_empty_sentences = True

# length of small debugging set to create
debug_length = 50

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    # summary dataframe to store counts of terms and sentences from each source
    summary_df = {
        'source': [],
        'data_split': [],
        'total_sentences': [],
        'sentences_with_term': [],
        'sentences_wo_overlap': [],
        'total_terms': [],
        'terms_wo_overlap': []
    }
    
    splits_data = {}
    for split, sources in splits.items():
        
        # count up terms for each split
        term_counts = Counter()
        
        # build up data for each split from all of its sources 
        sentences = []
        tags = []
        term_infos = [] 
        term_sources = []
        prev_length = 0
        for source in sources:
            
            with open(f"{input_data_dir}/{source}_tagged_sentences.json", 'r') as fid:
                tagged_source = json.load(fid)
                
            summary_df['source'].append(source)
            summary_df['data_split'].append(split)
            summary_df['total_sentences'].append(len(tagged_source))
            
            tmp_terms = []
            for sentence_data in tagged_source:
                tokenized_sentence = sentence_data['tokenized_text']
                tokenized_tags = sentence_data['tags']
                terms = sentence_data['found_terms']
                
                # ignore sentences with no tagged terms if specified
                if ignore_empty_sentences:
                    if len(set(tokenized_tags)) == 1:
                        continue
                
                # count up occurrences of each term
                for term in terms:
                    term_counts[term] += len(terms[term]['indices'])
                        
                sentences.append(' '.join(tokenized_sentence))
                tags.append(' '.join(tokenized_tags))
                term_sources.append(source)
                term_infos.append(terms)
                tmp_terms += list(terms.keys())
            
            # accumulate total sentences and term counts per source
            summary_df['sentences_with_term'].append(len(sentences) - prev_length)
            summary_df['total_terms'].append(len(set(tmp_terms)))
            prev_length = len(sentences)
                        
            
        splits_data[split] = {
            'sentences': sentences,
            'tags': tags,
            'terms': term_infos,
            'sources': term_sources,
            'term_counts': term_counts
        }
        
    # create set of terms to exclude from the training split 
    exclude_terms = []
    for split in train_exclude:
        exclude_terms += list(term_counts.keys())
    exclude_terms = set(exclude_terms)
        
    # remove sentences from the training set containing a term in an evaluation set
    filtered_train = {
        'sentences': [],
        'tags': [],
        'terms': [],
        'sources': [],
        'term_counts': Counter() 
    }
    for sentence, tag, term_info, source in zip(splits_data['train']['sentences'], 
                                                splits_data['train']['tags'], 
                                                splits_data['train']['terms'],
                                                splits_data['train']['sources']):
        sentences_terms = list(term_info.keys())
        overlap = [t in exclude_terms for t in sentences_terms]
        if not any(overlap):
            filtered_train['sentences'].append(sentence)
            filtered_train['tags'].append(tag)
            filtered_train['terms'].append(term_info)
            filtered_train['sources'].append(source)
            for term in term_info:
                filtered_train['term_counts'][term] += len(term_info[term]['indices'])
                
    splits_data['train'] = filtered_train
    
    # accumulate new sentence and term counts after adjusting for overlap
    for source, split in zip(summary_df['source'], summary_df['data_split']):
        summary_df['sentences_wo_overlap'].append( \
            len([s for s in splits_data[split]['sources'] if s == source]))
        summary_df['terms_wo_overlap'].append( \
            len(set([t for s, terms in zip(splits_data[split]['sources'],
                                           splits_data[split]['terms']) for t in terms
                     if s == source])))
    
    # create small debug set
    debug = {
        'sentences': splits_data['train']['sentences'][:debug_length],
        'tags': splits_data['train']['tags'][:debug_length],
        'terms': splits_data['train']['terms'][:debug_length],
        'sources': splits_data['train']['sources'][:debug_length]
    }
    splits_data['debug'] = debug
    
        
    # write out data splits
    for split in splits_data:
        with open(f"{output_data_dir}/term_extraction_{split}.json", 'w') as f:
            json.dump(splits_data[split], f, indent=4)
        
    # write out summary dataframe with source data counts
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(f"{summary_data_dir}/term_extraction_source_data_counts.csv", index=False)
    
    # write out aggregated data by split
    split_summary = {'split': [], 'sentences': [], 'terms': []}
    for split in splits:
        split_summary['split'].append(split)
        split_summary['sentences'].append(len(splits_data[split]['sentences']))
        split_summary['terms'].append(len(splits_data[split]['term_counts'].keys()))
    split_summary = pd.DataFrame(split_summary)
    split_summary.to_csv(f"{summary_data_dir}/term_extraction_split_data_counts.csv", index=False)
        