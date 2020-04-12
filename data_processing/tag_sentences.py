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
       
