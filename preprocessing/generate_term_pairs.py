# Enumerates all sentence, term pair combinations for a set of term-tagged sentences to be used
# for relation extraction
#
# Author: Matthew Boggess
# Version: 5/28/20
#
# Data Source: Tagged sentences dataframes output from tag_sentences.py.
#
# Description: 
#   For each provided set of tagged sentences: Enumerates all term pairs from the set of tagged 
#.  terms in each sentence creating a row in a new data frame for each pair.

#===================================================================================
# Libraries 

import spacy
import re
from data_processing_utils import get_closest_match
from tqdm import tqdm
import os
import pandas as pd
import json

#===================================================================================
# Parameters 

input_data_dir = '../data/preprocessed/tagged_sentences'
output_data_dir = '../data/preprocessed/term_pair_sentences'

textbooks = [
    'Life_Biology', 
    'Biology_2e'
]

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    for i, textbook in enumerate(textbooks):
        data = pd.read_pickle(f"{input_data_dir}/{textbook}_relation_extraction_tagged_sentences.pkl")
        
        new_df = []
        for k, row in tqdm(list(data.iterrows())):
            found_terms_info = row.term_info
            found_terms = list(found_terms_info.keys())
            found_term_pairs = [] 
            for i in range(len(found_terms) - 1):
                for j in range(i + 1, len(found_terms)):
                    term_pair = (found_terms[i], found_terms[j])
                    
                    indices = get_closest_match(
                        found_terms_info[term_pair[0]]['indices'],
                        found_terms_info[term_pair[1]]['indices']
                    )
                    
                    new_row = row.copy()
                    if indices[0][0] > indices[1][0]:
                        term_pair = (term_pair[1], term_pair[0])
                        indices = (indices[1], indices[0])
                    new_row['term_pair'] = term_pair
                    new_row['term1'] = term_pair[0]
                    new_row['term2'] = term_pair[1]
                    new_row['term1_location'] = indices[0]
                    new_row['term2_location'] = indices[1]
                    new_df.append(new_row)
                    
        data = pd.DataFrame(new_df).reset_index()
        data = data.drop(['index', 'tags', 'term_info'], axis=1)
        data.to_pickle(f"{output_data_dir}/{textbook}_term_pairs.pkl")
