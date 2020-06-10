# Tags sentences with all pairs of key terms from textbooks in preparation for 
# relation and term extraction modeling. 
#
# Author: Matthew Boggess
# Version: 5/28/20
#
# Data Source: Terms list output from collect_terms.py. Preprocessed sentences from
# preprocess_textbooks.py
#
# Description: 
#   For each provided textbook/textbook section: 
#     - Reads in term list and filters to desired term sources to use for tagging. Then tags
#       each sentence in the provided section with the term list.
#     - Saves the output in a pandas dataframe.
#   There are slightly different tagging approaches for term extraction and relation extraction
#   which can be switched between with the tag_type parameter.
#
# Running Note: This is a time intensive script. Tagging each individual textbook takes 
# on average 2-3 hours. 

#===================================================================================

# Libraries 

import spacy
import re
from data_processing_utils import tag_terms
from collections import Counter
from tqdm import tqdm
import os
import pandas as pd
import json

#===================================================================================

# Parameters

# whether to do the version of tagging used for relation_extraction or term_extraction
tag_type = 'relation_extraction'

## Filepaths

terms_file = "../data/preprocessed/terms/processed_terms.pkl"
sentences_dir = "../data/preprocessed/clean_sentences"
output_dir = '../data/preprocessed/tagged_sentences'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# invalid dependency parse tags that shouldn't be tagged
invalid_dep = ['npadvmod', 'compound', 'amod', 'nmod']
# whether to expand incomplete noun phrases when tagging
expand_np = False 

if tag_type == 'term_extraction':
    
    # invalid parts of speech that shouldn't be tagged
    invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP']
    
    # which term sources to use for tagging
    term_sources = [
        'Anatomy_and_Physiology',
        'Astronomy',
        'Chemistry_2e',
        'Microbiology',
        'Psychology',
        'University_Physics_Volume_1',
        'University_Physics_Volume_2',
        'University_Physics_Volume_3',
        'life_bio_ch39_hand_labelled',
        'openstax_bio2e_section10-2_hand_labelled',
        'openstax_bio2e_section10-4_hand_labelled',
        'openstax_bio2e_section4-2_hand_labelled',
    ] 
    
    # which textbooks we want to tag 
    textbooks = [
        'test',
        'dev',
        'Anatomy_and_Physiology',
        'Astronomy',
        'Chemistry_2e',
        'Microbiology',
        'Psychology',
        'University_Physics_Volume_1',
        'University_Physics_Volume_2',
        'University_Physics_Volume_3',
    ]
    
elif tag_type == 'relation_extraction':
    
    # invalid parts of speech that shouldn't be tagged
    invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 
                   'VBZ', 'VBP', 'WRB']
    
    term_sources = [
        'Biology_2e',
        'kb_bio101',
        'life_bio_ch39_hand_labelled',
        'openstax_bio2e_section10-2_hand_labelled',
        'openstax_bio2e_section10-4_hand_labelled',
        'openstax_bio2e_section4-2_hand_labelled',
    ]
    
    textbooks = [
        'Life_Biology', 
        'Biology_2e'
    ]

dev_term_sources = [
    'openstax_bio2e_section10-2_hand_labelled',
    'openstax_bio2e_section10-4_hand_labelled',
    'openstax_bio2e_section4-2_hand_labelled',
]

test_term_sources = ['life_bio_ch39_hand_labelled']



#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    # collect terms
    terms = pd.read_pickle(terms_file)
    train_terms = terms[terms.source.isin(term_sources)]
    train_terms = list(train_terms.drop_duplicates(['term']).doc)
    dev_terms = terms[terms.source.isin(dev_term_sources)]
    dev_terms = list(dev_terms.drop_duplicates(['term']).doc)
    test_terms = terms[terms.source.isin(test_term_sources)]
    test_terms = list(test_terms.drop_duplicates(['term']).doc)
    print(f"Collected {len(train_terms)} terms to use for tagging")

    for i, textbook in enumerate(textbooks):
    
        print(f"Tagging {textbook} textbook: {i + 1}/{len(textbooks)}")
        data = pd.read_pickle(f"{sentences_dir}/{textbook}_sentences.pkl")
        tokens = [] 
        annot = []
        term_info = []
        tags = []
        if textbook == 'dev':
            tagging_terms = dev_terms
            tagging_pos = []
            tagging_dep = []
        elif textbook == 'test':
            tagging_terms = test_terms
            tagging_pos = []
            tagging_dep = []
        else:
            tagging_terms = train_terms
            tagging_pos = invalid_pos
            tagging_dep = invalid_dep
            
        for _, row in tqdm(list(data.iterrows())): 
            result = tag_terms(row.doc, tagging_terms, nlp, invalid_pos=tagging_pos, 
                               invalid_dep=tagging_dep, expand_np=expand_np)
            tokens.append(result['tokenized_text'])
            tags.append(result['tags'])
            term_info.append(result['found_terms'])
            annot.append(result['annotated_text'])
        
        data['tokens'] = tokens
        data['tags'] = tags 
        data['term_info'] = term_info
        data['tagged_sentence'] = annot 
        data['terms'] = data.term_info.apply(lambda x: set(x.keys()))
            
        data.to_pickle(f"{output_dir}/{textbook}_{tag_type}_tagged_sentences.pkl")
