# Extracts individual sentences and key terms from each chapter of provided Openstax textbooks,
# preprocesses them using Stanford NLP pipeline, and saves the results for future use

# Author: Matthew Boggess
# Version: 4/3/20

# Data Source: Parsed textbook files of the openstax textbooks provided by openstax

# Description: 
#   For each specified openstax textbook: 
#     - extracts out individual sentences from the text of all chapters and runs them through
#       Stanford's NLP preprocessing pipeline including tokenization, pos tagging, lemmatization
#     - extracts out key terms from the key terms sections of each chapter and runs these
#       terms through the same Stanford NLP preprocessing pipeline

# Running Note: This is a time intensive script. Processing each individual textbook takes 30-40 
# minutes on average. It is best to subset the openstax_textbooks parameter to only textbooks that 
# need to be run if trying to add new ones or re-run particular textbooks.

#===================================================================================

# Libraries

import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
import pandas as pd
import re
import os
from tqdm import tqdm
import warnings
import json

#===================================================================================

# Parameters

## Filepaths

# data directories
raw_data_dir = "../data/raw_data/openstax/openstax_provided_book_import/sentence_files"
preprocessed_data_dir = "../data/preprocessed_data"
if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)

# special case: web scrape parse of microbiology key terms 
microbio_key_terms_file = f"../data/raw_data/openstax/openstax_webscrape/data/microbiology_glossary.json"

## Important Enumerations 

# Openstax textbooks to be processed
openstax_textbooks = [
    'Anatomy_and_Physiology', 
    'Astronomy', 
    'Biology_2e', 
    'Chemistry_2e',
    'Microbiology', 
    'Psychology', 
    'University_Physics_Volume_1',
    'University_Physics_Volume_2', 
    'University_Physics_Volume_3'
]

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

# Helper Functions

def parse_openstax_terms(key_term_text):
    """Parse openstax key term entries to extract key term itself (including acronyms)."""
    if ":" not in key_term_text:
        return []
    term = key_term_text.split(":")[0]
    match = re.match(".*\((.+)\).*", term)
    if match:
        acronym = match.group(1)
        term = term.replace(f"({acronym})", "")
        return [term.strip(), acronym.strip()]
    
    return [term.strip()] 

#===================================================================================

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    
    for i, textbook in enumerate(openstax_textbooks):
        
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)

        print(f"Processing {textbook} textbook: Textbook {i + 1}/{len(openstax_textbooks)}")
        textbook_data = pd.read_csv(f"{raw_data_dir}/sentences_{textbook}_parsed.csv")
        
        # spacy preprocess sentences
        print("Running chapter sentences through Stanford NLP pipeline")
        output_file = f"{preprocessed_data_dir}/{textbook}_sentences_spacy"
        output_vocab_file = f"{preprocessed_data_dir}/{textbook}_sentences_spacy_vocab"
        sentences = textbook_data[~textbook_data.section_name.isin(exclude_sections)].sentence
        sentences_spacy = []
        for sent in tqdm(sentences):
            if not len(sent):
                continue
            sentences_spacy.append(nlp(sent))
        write_spacy_docs(sentences_spacy, nlp.vocab, output_file, output_vocab_file)
        
        # spacy preprocess key terms
        print("Running chapter key terms through Stanford NLP pipeline")
        output_file = f"{preprocessed_data_dir}/{textbook}_key_terms_spacy"
        output_vocab_file = f"{preprocessed_data_dir}/{textbook}_key_terms_spacy_vocab"
        key_terms_spacy = []
        # use special extracted key terms file for microbiology since key term format is different
        if textbook == "Microbiology":
            with open(microbio_key_terms_file, "r") as f:
                key_terms = json.load(f)
            for key_term in tqdm(key_terms):
                key_terms_spacy.append(nlp(key_term))
                acronym = key_terms[key_term]["acronym"]
                if len(acronym):
                    key_terms_spacy.append(nlp(acronym))
        else:
            key_terms = textbook_data[textbook_data.section_name == "Key Terms"].sentence
            for key_term in tqdm(key_terms):
                kts = parse_openstax_terms(key_term)
                if len(kts):
                    key_terms_spacy += [nlp(kt) for kt in kts]
        write_spacy_docs(key_terms_spacy, nlp.vocab, output_file, output_vocab_file)
        
