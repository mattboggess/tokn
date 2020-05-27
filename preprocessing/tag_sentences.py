# Tags sentences with all pairs of key terms from textbooks in preparation for 
# relation extraction modeling. 

# Author: Matthew Boggess
# Version: 4/26/20

# Data Source: Terms list output from collect_terms.py. Raw sentence parses of OpenStax Biology
# (provided by OpenStax) and Life Biology (provided by Dr. Chaudhri)

# Description: 
#   For each provided textbook/textbook section: 
#     - Reads in spacy preprocessed terms from all specified term sources that are indicated to use
#       for tagging and aggregates them into a term list to tag the textbook/section sentences. This
#       includes optional filtering on part of speech.
#     - Creates a pandas dataframe for each textbook that takes every tagged pair of terms in that
#       sentence and creates two rows for each (one for each direction of the term pair). Each row
#       includes the following information:
#          - sentence textbook source, chapter, and section identifier (together these uniquely
#            identify each sentence)
#          - original sentence text and a list of Spacy tokens
#          - term1/2 Spacy lemma representation and their indices in the token list
#
# Running Note: This is a time intensive script. Processing each individual textbook takes 
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

## Filepaths

terms_file = "../data/preprocessed/terms/processed_terms.pkl"
sentences_dir = "../data/preprocessed/clean_sentences"
output_dir = '../data/preprocessed/tagged_sentences'

# invalid parts of speech that shouldn't be tagged
invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP']

# invalid dependency parse tags that shouldn't be tagged
invalid_dep = ['npadvmod', 'compound', 'poss', 'amod', 'nmod']

# text representations of concepts that are too general and thus problematic for text matching
exclude_terms = ['object', 'aggregate', 'group', 'thing', 'region', 'center', 'response',
                 'series', 'unit', 'result', 'normal', 'divide', 'whole', 'someone', 'somebody',
                 'feature', 'class', 'end', 'lead', 'concept', 'present', 'source', 'event', 'limit']

expand_np = False 

term_sources = [
    #'Anatomy_and_Physiology',
    #'Astronomy',
    'Biology_2e',
    #'Chemistry_2e',
    #'Microbiology',
    #'Psychology',
    'kb_bio101',
    #'University_Physics_Volume_1',
    #'University_Physics_Volume_2',
    #'University_Physics_Volume_3',
    'life_bio_ch39_hand_labelled',
    'openstax_bio2e_section10-2_hand_labelled',
    'openstax_bio2e_section10-4_hand_labelled',
    'openstax_bio2e_section4-2_hand_labelled',
] 

dev_term_sources = [
    'openstax_bio2e_section10-2_hand_labelled',
    'openstax_bio2e_section10-4_hand_labelled',
    'openstax_bio2e_section4-2_hand_labelled',
]

test_term_sources = ['life_bio_ch39_hand_labelled']

textbooks = [
    'test',
    'dev',
    'Life_Biology', 
    'Biology_2e',
    'Anatomy_and_Physiology',
    'Astronomy',
    'Chemistry_2e',
    'Microbiology',
    'Psychology',
    'University_Physics_Volume_1',
    'University_Physics_Volume_2',
    'University_Physics_Volume_3',
]

textbooks = [
    'test',
    'dev',
    'Life_Biology', 
    'Biology_2e'
]

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
    #term_files = os.listdir(terms_dir)
    #seen_lemmas = set()
    #terms = []
    #dev_terms = []
    #test_terms = []
    #for tf in term_files:
    #    valid = re.match('(.*)_terms.pkl', tf)
    #    if not valid:
    #        continue
    #    source = valid.group(1)
    #    if source not in term_sources:
    #        continue
    #    t = pd.read_pickle(f"{terms_dir}/{tf}")
    #    for doc in t.doc:
    #        term_lemma = ' '.join([x.lemma_ for x in doc])
    #        if source in dev_term_sources:
    #            dev_terms.append(doc)
    #        if source in test_term_sources:
    #            test_terms.append(doc)
    #        if term_lemma in seen_lemmas or term_lemma in exclude_terms:
    #            continue
    #        terms.append(doc)
    print(f"Collected {len(train_terms)} terms to use for tagging")

    files = os.listdir(sentences_dir)
    for i, file in enumerate(files):
    
        valid = re.match('(.*)_sentences.pkl', file)
        if not valid:
            continue
        textbook = valid.group(1)
        if textbook not in textbooks:
            continue
        
        print(f"Tagging {textbook} textbook: {i + 1}/{len(files)}")
        data = pd.read_pickle(f"{sentences_dir}/{file}")
        tokens = [] 
        annot = []
        term_info = []
        tags = []
        if textbook == 'dev':
            tagging_terms = dev_terms
        elif textbook == 'test':
            tagging_terms = test_terms
        else:
            tagging_terms = train_terms
        for _, row in tqdm(list(data.iterrows())): 
            result = tag_terms(row.doc, tagging_terms, nlp, invalid_pos=invalid_pos, 
                               invalid_dep=invalid_dep, expand_np=expand_np)
            tokens.append(result['tokenized_text'])
            tags.append(result['tags'])
            term_info.append(result['found_terms'])
            annot.append(result['annotated_text'])
        
        data['tokens'] = tokens
        data['tags'] = tags 
        data['term_info'] = term_info
        data['tagged_sentence'] = annot 
        data['terms'] = data.term_info.apply(lambda x: set(x.keys()))
            
        data.to_pickle(f"{output_dir}/{textbook}_tagged_sentences.pkl")
