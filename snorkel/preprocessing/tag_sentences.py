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
from data_processing_utils import tag_terms, read_spacy_docs, get_closest_match
from collections import Counter
from tqdm import tqdm
import os
import pandas as pd
import json

#===================================================================================

# Parameters

## Filepaths

terms_file = "../data/biology_terms_spacy"
terms_vocab_file = "../data/biology_terms_spacy_vocab"

life_bio_input_sentences_file = "../../data/raw_data/life_bio/life_bio_selected_sentences.txt"
life_bio_output_file = "../data/life_bio_tagged_sentences.csv"
openstax_bio_input_sentences_file = "../../data/raw_data/openstax/openstax_provided_book_import/sentence_files/sentences_Biology_2e_parsed.csv"
openstax_bio_output_file = "../data/openstax_bio_tagged_sentences.csv"
    
# Life Biology Section Regex
life_bio_regex = '^7\.(\d+)\.(.+?)\s+'

# Valid part of speech tags to be considered for tagging terms
valid_pos = ['NOUN', 'PROPN', 'ADJ']

# textbook sections to exclude from consideration when extracting chapter sentences 
exclude_sections = [
    'Preface',
    'Chapter Outline',
    'Index',
    'Chapter Outline',
    'Summary',
    'Multiple Choice',
    'Fill in the Blank',
    'short Answer',
    'Critical Thinking',
    'References',
    'Units',
    'Conversion Factors',
    'Fundamental Constants',
    'Astronomical Data',
    'Mathematical Formulas',
    'The Greek Alphabet',
    'Chapter 1',
    'Chapter 2',
    'Chapter 3',
    'Chapter 4',
    'Chapter 5',
    'Chapter 6',
    'Chapter 7',
    'Chapter 8'
    'Chapter 9',
    'Chapter 10',
    'Chapter 11',
    'Chapter 12',
    'Chapter 13',
    'Chapter 14',
    'Chapter 15',
    'Chapter 16',
    'Chapter 17',
    'Critical Thinking Questions',
    'Visual Connection Questions', 
    'Key Terms', 
    'Review Questions', 
    'Glossary',
    'The Periodic Table of Elements', 
    'Measurements and the Metric System'
]


#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    # load in biology terms to use for tagging
    terms = read_spacy_docs(terms_file, terms_vocab_file)
    print(f"Found {len(terms)} candidate biology terms to use for tagging")
    
    # filter out invalid terms with wrong POS
    terms = [term for term in terms if term[-1].pos_ in valid_pos]
    print(f"{len(terms)} biology terms remaining for tagging after POS filtering")
    
    life_bio_df = []
    
    print("Processing Life Biology Sentences")
    with open(life_bio_input_sentences_file, "r") as f:
        life_bio_sentences = f.readlines()
        
    prev_section = None
    for sent in tqdm(life_bio_sentences):
        
        # parse chapter and section
        sections = re.match(life_bio_regex, sent)
        chapter = sections.group(1)
        sent_id = sections.group(2)
        sent = re.sub(life_bio_regex, '', sent.strip())
        
        # Spacy process and tag sentence with terms
        spacy_sent = nlp(sent)
        result = tag_terms(spacy_sent, terms, nlp)
        found_terms_info = result['found_terms']
        tokenized_sent = result['tokenized_text']
        
        # generate all term pairs tagged in the sentence, for multiple mentions take pair of
        # terms that are closest together
        found_terms = list(found_terms_info.keys())
        found_term_pairs = [] 
        for i in range(len(found_terms) - 1):
            for j in range(i + 1, len(found_terms)):
                term_pair = (found_terms[i], found_terms[j])
                indices = get_closest_match(
                    found_terms_info[term_pair[0]]['indices'],
                    found_terms_info[term_pair[1]]['indices']
                )
                found_term_pairs.append((term_pair, indices))
                
                term_pair_reverse = (found_terms[j], found_terms[i])
                indices_reverse = get_closest_match(
                    found_terms_info[term_pair_reverse[0]]['indices'],
                    found_terms_info[term_pair_reverse[1]]['indices']
                )
                found_term_pairs.append((term_pair_reverse, indices_reverse))
        
        for found_pair in found_term_pairs:
            life_bio_df.append(
                {'textbook': 'Life Biology',
                 'chapter': chapter,
                 'sentence_id': sent_id,
                 'text': sent,
                 'tokens': tokenized_sent,
                 'term1': found_pair[0][0],
                 'term1_location': found_pair[1][0],
                 'term2': found_pair[0][1],
                 'term2_location': found_pair[1][1],
                 'term_pair': (found_pair[0][0], found_pair[1][0])
                }
            )
            
    life_bio_df = pd.DataFrame(life_bio_df)
    life_bio_df.to_csv(life_bio_output_file, index=False)

    print("Processing OpenStax Biology Sentences")
    openstax_data = pd.read_csv(openstax_bio_input_sentences_file)
    openstax_data = openstax_data[~openstax_data.section_name.isin(exclude_sections)]
    
    openstax_bio_df = []
    prev_section = None
    for _, row in tqdm(openstax_data.iterrows()):
        
        # parse chapter and section
        chapter = row.chapter
        sent_id = f'{row.section}.{row.sentence_number}'
        sent = row.sentence.strip()
        
        # Spacy process and tag sentence with terms
        spacy_sent = nlp(sent)
        result = tag_terms(spacy_sent, terms, nlp)
        found_terms_info = result['found_terms']
        tokenized_sent = result['tokenized_text']
        
        # generate all term pairs tagged in the sentence, for multiple mentions take pair of
        # terms that are closest together
        found_terms = list(found_terms_info.keys())
        found_term_pairs = [] 
        for i in range(len(found_terms) - 1):
            for j in range(i + 1, len(found_terms)):
                term_pair = (found_terms[i], found_terms[j])
                indices = get_closest_match(
                    found_terms_info[term_pair[0]]['indices'],
                    found_terms_info[term_pair[1]]['indices']
                )
                found_term_pairs.append((term_pair, indices))
                
                term_pair_reverse = (found_terms[j], found_terms[i])
                indices_reverse = get_closest_match(
                    found_terms_info[term_pair_reverse[0]]['indices'],
                    found_terms_info[term_pair_reverse[1]]['indices']
                )
                found_term_pairs.append((term_pair_reverse, indices_reverse))
        
        for found_pair in found_term_pairs:
            openstax_bio_df.append(
                {'textbook': 'OpenStax Biology 2e',
                 'chapter': chapter,
                 'sentence_id': sent_id,
                 'text': sent,
                 'tokens': tokenized_sent,
                 'term1': found_pair[0][0],
                 'term1_location': found_pair[1][0],
                 'term2': found_pair[0][1],
                 'term2_location': found_pair[1][1],
                 'term_pair': (found_pair[0][0], found_pair[1][0])
                }
            )
            
    openstax_bio_df = pd.DataFrame(openstax_bio_df)
    openstax_bio_df.to_csv(openstax_bio_output_file, index=False)