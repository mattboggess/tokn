# Preprocesses individual section/chapter sentences and key term list using Spacy NLP pipeline
# for sections/chapters of textbooks that have had key term lists manually extraced.

# Author: Matthew Boggess
# Version: 4/11/20

# Data Source: Manually copied text files of particular textbook sections/chapters and manually
# constructed key terms lists compiled by subject matter experts.

# Description: 
#   For each specified textbook section/chapter: 
#     - extracts out individual sentences from the text of that section/chapter and runs them '
#       through Spacy's NLP preprocessing pipeline including tokenization, pos tagging, lemmatization
#     - runs a manually curated key term list through the same Spacy NLP preprocessing pipeline

#===================================================================================

# Libraries

import spacy
from nltk.tokenize import sent_tokenize
from data_processing_utils import write_spacy_docs
import pandas as pd
import re
import os
from tqdm import tqdm
import json

#===================================================================================

# Parameters

## Filepaths

# data directories
raw_data_dir = "../data/raw_data/hand_labeled"
preprocessed_data_dir = "../data/preprocessed_data"
if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)

## Important Enumerations 

# textbook sections/chapters to be processed
textbook_sections = [
    'openstax_bio2e_section10-2',
    'openstax_bio2e_section10-4',
    'openstax_bio2e_section4-2'
]

#===================================================================================

if __name__ == "__main__":
    
    for i, section in enumerate(textbook_sections):
        
        nlp = spacy.load("en_core_web_sm")

        print(f"Processing {section} textbook section: Section {i + 1}/{len(textbook_sections)}")
        with open(f"{raw_data_dir}/{section}_text.txt", 'r') as fid:
            section_text = fid.read()
        
        # spacy preprocess sentences
        print("Running section sentences through Spacy NLP pipeline")
        output_file = f"{preprocessed_data_dir}/{section}_sentences_spacy"
        output_vocab_file = f"{preprocessed_data_dir}/{section}_sentences_spacy_vocab"
        section_sentences = sent_tokenize(section_text)
        sentences_spacy = []
        for sent in tqdm(section_sentences):
            if not len(sent):
                continue
            sentences_spacy.append(nlp(sent))
        write_spacy_docs(sentences_spacy, nlp.vocab, output_file, output_vocab_file)
        
        # spacy preprocess key terms
        print("Running section terms through Spacy NLP pipeline")
        output_file = f"{preprocessed_data_dir}/{section}_key_terms_spacy"
        output_vocab_file = f"{preprocessed_data_dir}/{section}_key_terms_spacy_vocab"
        key_terms_spacy = []
        # use special extracted key terms file for microbiology since key term format is different
        with open(f"{raw_data_dir}/{section}_terms.txt", 'r') as fid:
            key_terms = fid.readlines()
            for key_term in tqdm(key_terms):
                key_term = nlp(key_term.split('--')[0].strip())
                if len(key_term):
                    key_terms_spacy.append(key_term) 
        write_spacy_docs(key_terms_spacy, nlp.vocab, output_file, output_vocab_file)
