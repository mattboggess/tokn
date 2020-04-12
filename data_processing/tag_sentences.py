# Tags sentences with key terms from textbooks in preparation for term extraction modeling. 

# Author: Matthew Boggess
# Version: 4/11/20

# Data Source: Output of preprocess_openstax_textbooks.py, preprocess_life_biology.py, and
# preprocess_hand-labeled_sections.py scripts.

# Description: 
#   For each provided textbook/textbook section: 
#     - Reads in spacy preprocessed terms from all specified term sources and aggregates them
#       all into a single list of terms to tag the textbooks with (special exception is that
#       the life biology kb is tagged on its own)
#     - Produces three output files for each textbook:
#         1. text file with tokenized sentences from the textbook where each token is separated
#            by a space (one sentence per line)
#         2. text file with NER BIOES tags for each token in the tokenized sentences separated
#            by a space (one sentence per line)
#         3. json file with term counts denoting how many times each term was tagged across all
#            sentences in the textbook

# Running Note: This is a very time intensive script. Processing each individual textbook takes 
# on average 3 hours. It is best to subset the textbooks parameter to only textbooks that 
# need to be run if trying to add new ones or re-run particular textbooks.

#===================================================================================

# Libraries 

import spacy
from data_processing_utils import tag_terms, read_spacy_docs
from collections import Counter
import warnings
from tqdm import tqdm
import os
import json

#===================================================================================

# Parameters

## Filepaths

input_data_dir = "../data/preprocessed_data"
output_data_dir = "../data/term_extraction/tagged_sentences"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

## Other parameters 

# list of all term sources we want to combine to create a large list for tagging
# we exclude the Life Biology KB terms from this aggregate since the nature of the
# kb leads to many generic descriptions of terms that we don't want to generally tag (i.e. group)
pooled_terms = [
    'Anatomy_and_Physiology', 
    'Astronomy', 
    'Biology_2e',
    'Chemistry_2e',
    'Microbiology',
    'Psychology',
    'University_Physics_Volume_1',
    'University_Physics_Volume_2',
    'University_Physics_Volume_3',
    'openstax_bio2e_section10-2'
]

# mapping from textbooks/textbook sections whose sentences we would like to tag to the
# sources of terms we should use for tagging them
textbooks = {
    'openstax_bio2e_section10-2': ['openstax_bio2e_section10-2']
    'Anatomy_and_Physiology': pooled_terms, 
    'Astronomy': pooled_terms, 
    'Biology_2e': pooled_terms,
    'Chemistry_2e': pooled_terms,
    'Life_Biology_kb': ['Life_Biology_kb'],
    'Microbiology': pooled_terms,
    'Psychology': pooled_terms,
    'University_Physics_Volume_1': pooled_terms,
    'University_Physics_Volume_2': pooled_terms,
    'University_Physics_Volume_3': pooled_terms
}

#===================================================================================

if __name__ == '__main__':
    
    #warnings.filterwarnings('ignore')
    nlp = spacy.load("en_core_web_sm")
    
    for i, (textbook, term_sources) in enumerate(textbooks.items()):
        print(f"Tagging {textbook} sentences: Input {i + 1}/{len(textbooks)}")
        
        terms = []
        for term_source in term_sources:
            terms += read_spacy_docs(
                f"{input_data_dir}/{term_source}_key_terms_spacy",
                f"{input_data_dir}/{term_source}_key_terms_spacy_vocab"
            )
    
        tagged_sentences = []
        spacy_sentences = read_spacy_docs(
            f"{input_data_dir}/{textbook}_sentences_spacy",
            f"{input_data_dir}/{textbook}_sentences_spacy_vocab"
        )
        for sentence in tqdm(spacy_sentences):
            
            result = tag_terms(sentence, terms, nlp)
            result['original_text'] = sentence.text_with_ws
            tagged_sentences.append(result)
            
        with open(f"{output_data_dir}/{textbook}_tagged_sentences.json", 'w') as fid:
            json.dump(tagged_sentences, fid, indent=4)
       
