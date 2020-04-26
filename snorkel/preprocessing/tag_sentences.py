# Tags sentences with key terms from textbooks in preparation for term extraction modeling. 

# Author: Matthew Boggess
# Version: 4/11/20

# Data Source: Output of preprocess_openstax_textbooks.py, preprocess_life_biology.py, and
# preprocess_hand-labeled_sections.py scripts.

# Description: 
#   For each provided textbook/textbook section: 
#     - Reads in spacy preprocessed terms from all specified term sources that are indicated to use
#       for tagging and aggregates them into a term lis to tag the textbook/section sentences 
#     - Produces a single json output file for each textbook/section containing an entry with
#       the following for each sentence:
#       - found_terms: structure indicating what terms were found and some information about
#         their matche(s) in the sentence
#       - tokenized_text: list of individual tokens from the tokenization of the sentence. This is 
#         the input to the term extraction models.
#       - tags: list of BIOES tags for each individual token in the tokenized sentence. This is
#         the label for training the model and is the same form as the output.
#       - annotated_text: This is the string representation of the sentence with special indicators
#         denoting where the found terms are located.
#       - original_text: Original string representation of the sentence for reference.

# Running Note: This is a very time intensive script. Processing each individual textbook takes 
# on average 3 hours. It is best to subset the textbooks parameter to only textbooks that 
# need to be run if trying to add new ones or re-run particular textbooks.

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
life_bio_regex = '^7\.(\d+)\.([\d*|summary])*\.*.+\..+?\s*'

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
    
    life_bio_df = []
    
    print("Processing Life Biology Sentences")
    with open(life_bio_input_sentences_file, "r") as f:
        life_bio_sentences = f.readlines()
        
    prev_section = None
    for sent in tqdm(life_bio_sentences):
        
        # parse chapter and section
        sections = re.match(life_bio_regex, sent)
        chapter = sections.group(1)
        section = sections.group(2)
        if not section:
            section = 'intro'
        if not prev_section or prev_section != section:
            prev_section = section
            sent_num = 1
        else:
            sent_num += 1
        sent = re.sub(life_bio_regex, '', sent.strip())
        
        # nlp process and tag sentence
        spacy_sent = nlp(sent)
        result = tag_terms(spacy_sent, terms, nlp)
        found_terms_info = result['found_terms']
        tokenized_sent = result['tokenized_text']
        
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
                 'section': section,
                 'sentence': sent_num,
                 'text': sent,
                 'tokens': tokenized_sent,
                 'term1': found_pair[0][0],
                 'term1_location': found_pair[1][0],
                 'term2': found_pair[0][1],
                 'term2_location': found_pair[1][1]
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
        section = row.section
        sent = row.sentence.strip()
        if not prev_section or prev_section != section:
            prev_section = section
            sent_num = 1
        else:
            sent_num += 1
        
        # nlp process and tag sentence
        spacy_sent = nlp(sent)
        result = tag_terms(spacy_sent, terms, nlp)
        found_terms_info = result['found_terms']
        tokenized_sent = result['tokenized_text']
        
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
                 'section': section,
                 'sentence': sent_num,
                 'text': sent,
                 'tokens': tokenized_sent,
                 'term1': found_pair[0][0],
                 'term1_location': found_pair[1][0],
                 'term2': found_pair[0][1],
                 'term2_location': found_pair[1][1]
                }
            )
            
    openstax_bio_df = pd.DataFrame(openstax_bio_df)
    openstax_bio_df.to_csv(openstax_bio_output_file, index=False)