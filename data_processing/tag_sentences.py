# Tags sentences with key terms from textbooks in preparation for term extraction modeling. 

# Author: Matthew Boggess
# Version: 4/3/20

# Data Source: Output of preprocess_openstax_textbooks.py and preprocess_life_biology.py
# scripts.

# Description: 
#   For each provided textbook: 
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

import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
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

# textbooks whose sentences we would like to tag
textbooks = [
    'Anatomy_and_Physiology', 
    'Astronomy', 
    'Biology_2e',
    'Chemistry_2e',
    'Life_Biology_kb',
    'Microbiology',
    'Psychology',
    'University_Physics_Volume_1',
    'University_Physics_Volume_2',
    'University_Physics_Volume_3'
]

# textbooks whose key terms should be aggregated together to create a full term list for tagging 
term_sources = [
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

#===================================================================================

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    
    # aggregate key terms into single list
    agg_terms = []
    for term_source in term_sources:
        agg_terms += read_spacy_docs(
            f"{input_data_dir}/{term_source}_key_terms_spacy",
            f"{input_data_dir}/{term_source}_key_terms_spacy_vocab"
        )
    
    for i, textbook in enumerate(textbooks):
        print(f"Tagging {textbook} textbook sentences: Textbook {i + 1}/{len(textbooks)}")
        
        # special case handling of the life bio knowledge base, we use it specifically for
        # evaluation, but don't include its terms generally since there are many too general
        # word representations for bio concepts in the knowledge base
        if textbook == 'Life_Biology_kb':
            terms = read_spacy_docs(
                f"{input_data_dir}/Life_Biology_kb_key_terms_spacy",
                f"{input_data_dir}/Life_Biology_kb_key_terms_spacy_vocab"
            )
        else:
            terms = agg_terms
    
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
       
