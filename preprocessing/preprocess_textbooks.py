# Filters parsed sentences from OpenStax textbooks and Life Biology to only relevant chapter 
# sentences and Spacy preprocesses the sentences. Also, pulls out key terms from the textbooks.
#
# Author: Matthew Boggess
# Version: 5/28/20

# Data Source: Parsed csv sentences of Life Biology and OpenStax textbooks from parsing scripts 

# Description: 
#   For each textbook, filters sentences only to educational sections. Then spacy preprocesses each
#   sentence. Saved to pickled pandas dataframe (with spacy doc) and csv (without spacy doc).
#   Pulls out key term lists from each textbook and saves them to newline delimited text files.

#===================================================================================
# Libraries 

import os
import pandas as pd
import spacy
import re
from tqdm import tqdm

#===================================================================================
# Helper Functions 

def parse_openstax_terms(key_term_text):
    """Parse openstax key term entries to extract key term itself (including acronyms)."""
    key_term_text = re.sub('<.+?>', '', key_term_text)
    key_term_text = key_term_text.replace('-', ' ').replace('\n', '').replace('\s+', ' ')
    if ":" not in key_term_text:
        return []
    term = key_term_text.split(':')[0]
    match = re.match('.*\((.+)\).*', term)
    if match:
        acronym = match.group(1)
        term = term.replace(f"({acronym})", "")
        return [term.strip(), acronym.strip()]
    
    return [term.strip()] 

#===================================================================================
# Parameters 

sent_input = '../data/preprocessed/parsed_books/'
sent_output = '../data/preprocessed/clean_sentences/'
term_output = '../data/preprocessed/terms/'

# chapters/sections to be separated into a dev/test set for 
test = ('Life_Biology', [([39], [0, 1, 2, 3, 4, 5, 6])])
dev = ('Biology_2e', [([4], [2]), ([10], [2, 4])])

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

# sections that hold key terms
term_sections = ['Key Terms', 'Index']

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')

    files = os.listdir(sent_input)

    for i, file in enumerate(files):

        textbook = re.match('sentences_(.*)_parsed.csv', file)
        if not textbook:
            continue
        textbook = textbook.group(1)

        print(f"Processing {textbook} textbook: {i + 1}/{len(files)}")

        book = pd.read_csv(f"{sent_input}/{file}")
        book['textbook'] = textbook

        # process terms into newline separated text file 
        print("Processing Terms")
        terms = book[book.section_name.isin(term_sections)].sentence
        terms = [t.split(':')[0] for t in terms]
        new_terms = []
        for term in terms:
            term = re.split('\(|\)', term)
            if len(term) == 1:
                new_terms.append(term[0].strip())
            elif len(term) == 3:
                new_terms.append((term[0].strip() + ' ' + term[2].strip()).strip())
                new_terms.append((term[1].strip() + ' ' + term[2].strip()).strip())
            else:
                pass
        terms = list(set(new_terms))
        terms = [re.sub(' *- *', ' ', t) for t in terms]
        terms = sorted([t for t in terms if not t.startswith('of ')])
        if terms:
            with open(f"{term_output}/{textbook}_terms.txt", 'w') as fid:
                fid.write('\n'.join(terms))

        # spacy preprocess relevant sentences and save 
        print("Processing Sentences")
        book = book[~book.section_name.isin(exclude_sections)]
        docs = []
        for sent in tqdm(book.sentence):
            docs.append(nlp(sent))
        book['doc'] = docs 
        if 'page_id' in book.columns:
            book = book.drop(['page_id'], axis=1)
        book.to_pickle(f"{sent_output}/{textbook}_sentences.pkl")
        book.drop(['doc'], axis=1).to_csv(f"{sent_output}/{textbook}_sentences.csv", index=False)

        # special separate versions for dev and test sets for term extraction
        if textbook == dev[0]:
            textbook = 'dev'
            split = dev
        elif textbook == test[0]:
            textbook = 'test'
            split = test 
        else:
            continue
        new_book = []
        for ch, sects in split[1]:
            new_book.append(book[(book.chapter.isin(ch)) & (book.section.isin(sects))])
        book = pd.concat(new_book)
    
        book.to_pickle(f"{sent_output}/{textbook}_sentences.pkl")
        book.drop(['doc'], axis=1).to_csv(f"{sent_output}/{textbook}_sentences.csv", index=False)

        
