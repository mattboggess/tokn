# Libraries 

import spacy
import re
from data_processing_utils import get_closest_match
from tqdm import tqdm
import os
import pandas as pd
import json

input_data_dir = '../data/preprocessed/tagged_sentences'
output_data_dir = '../data/preprocessed/term_pair_sentences'

textbooks = [
    'Life_Biology', 
    'Biology_2e'
    #'Anatomy_and_Physiology',
    #'Astronomy',
    #'Chemistry_2e',
    #'Microbiology',
    #'Psychology',
    #'University_Physics_Volume_1',
    #'University_Physics_Volume_2',
    #'University_Physics_Volume_3',
]

# invalid parts of speech that shouldn't be tagged
invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 
               'VBP', 'WRB']

one_direction = True

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    for i, textbook in enumerate(textbooks):
        data = pd.read_pickle(f"{input_data_dir}/{textbook}_tagged_sentences.pkl")
        
        new_df = []
        for k, row in tqdm(list(data.iterrows())):
            found_terms_info = row.term_info
            found_terms = list(found_terms_info.keys())
            found_term_pairs = [] 
            for i in range(len(found_terms) - 1):
                for j in range(i + 1, len(found_terms)):
                    term_pair = (found_terms[i], found_terms[j])
                    
                    if found_terms_info[term_pair[0]]['pos'][0][-1] in invalid_pos or \
                       found_terms_info[term_pair[1]]['pos'][0][-1] in invalid_pos:
                        continue
                        
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

                    #if not one_direction:
                    #    term_pair_reverse = (found_terms[j], found_terms[i])
                    #    indices_reverse = get_closest_match(
                    #        found_terms_info[term_pair_reverse[0]]['indices'],
                    #        found_terms_info[term_pair_reverse[1]]['indices']
                    #    )
                    #    found_term_pairs.append((term_pair_reverse, indices_reverse))
        data = pd.DataFrame(new_df).reset_index()
        data = data.drop(['level_0', 'index'], axis=1)
        data.to_pickle(f"{output_data_dir}/{textbook}_term_pairs.pkl")
