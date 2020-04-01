# Extracts individual sentences and key terms from each chapter of provided Openstax textbooks
# and preprocesses them using Spacy NLP pipeline and saves the results for future use

# Author: Matthew Boggess
# Version: 4/1/20

# Data Source: Partially parsed textbook files of the openstax textbooks were provided by openstax

# Description: 
#   For each specified openstax textbook: 
#     - extracts out individual sentences from the text of all chapters and runs them through
#       Spacy's preprocessing pipeline including tokenization, pos tagging, lemmatization, etc.
#     - extracts out key terms from the key terms sections of each chapter and runs these
#       terms through the same Spacy preprocessing pipeline

#===================================================================================

# Libraries

import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
import warnings
from io import StringIO
import pandas as pd
import re
import tqdm
import json

#===================================================================================

# Parameters

## Filepaths

# data directories
raw_data_dir = "../data/raw_data/life_bio"
preprocessed_data_dir = "../data/preprocessed_data"
if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)

# lexicon/kb input and output files for life biology
lexicon_input_file = f"{raw_data_dir}/kb_lexicon.txt"
lexicon_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_lexicon.json"
bio_concepts_file = f"{raw_data_dir}/kb_biology_concepts.txt"
terms_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_key_terms_spacy"

# sentence input and output files for life biology
input_sentences_file = f"{raw_data_dir}/life_bio_selected_sentences.txt"
kb_output_sentences_file = f"{preprocessed_data_dir}/Life_Biology_kb_sentences_spacy"
output_sentences_file = f"{preprocessed_data_dir}/Life_Biology_sentences_spacy"

## Important Enumerations 

# 
STOP_WORDS = ['object', 'aggregate', 'group', 'thing', 'region']

#===================================================================================
# Helper Functions

def parse_openstax_terms(key_term_text):
    """ Parse openstax data to extract key terms (including acronyms). """
    if ":" not in key_term_text:
        return []
    term = key_term_text.split(":")[0]
    match = re.match(".*\((.+)\).*", term)
    if match:
        acronym = match.group(1)
        term = term.replace(f"({acronym})", "")
        return [term.strip(), acronym.strip()]
    
    return [term.strip()] 

def process_lexicon(lexicon, bio_concepts):
    """ Takes in a lexicon consisting of concept text representation pairs and turns this into a 
    list of Spacy processed terms and a lexicon csv mapping KB concepts to lists of text 
    representations and their lemma forms.
    """
    
    # get rid of extra column and read in as dataframe
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, 
                          names=['concept', 'relation', 'text', 'pos'])
    
    concept_types = lexicon.query("text in ['Entity', 'Event']")
    lexicon = lexicon.query("text not in ['Entity', 'Event']")

    # create mapping from kb concept to unique text representations
    lexicon = lexicon[~lexicon.text.str.contains('Concept-Word-Frame')]
    lexicon = lexicon.groupby('concept')['text'].apply(
        lambda x: list(set([t for t in x if t not in STOP_WORDS]))).reset_index()
    
    # filter out too general upper ontology words, relation concepts, and 
    # concepts that only have stop words 
    lexicon = lexicon[lexicon.concept.isin(bio_concepts)]
    lexicon = lexicon[lexicon.text.map(len) > 0]
    lexicon = lexicon[lexicon.text.apply(lambda x: 'Relation' not in x)]

    # spacy process terms to get lemmas
    spacy_terms = []
    lexicon_output = {}
    for concept in lexicon.concept:
        
        # extract text representations for the concept
        terms = list(lexicon.loc[lexicon.concept == concept, 'text'])[0]
        terms = [t.replace('"', '').strip() for t in terms]
        
        # add text of concept itself
        concept_text = concept.lower().replace('-', ' ')
        if concept_text not in terms:
            terms.append(concept_text)
        
        # spacy preprocess and add lemmas
        spacy_terms_tmp = [nlp(term) for term in terms]
        lemma_terms = [' '.join([tok.lemma_ for tok in t]) for t in spacy_terms_tmp]
        
        # accumulate spacy term representations
        spacy_terms += spacy_terms_tmp
        
        # add concept entry to output lexicon
        lexicon_output[concept] = {
            'text_representations': terms,
            'class_label': concept_types.loc[concept_types.concept == concept, 'text'].iat[0],
            'lemma_representations': list(set(lemma_terms))
        }

    # filter out upper ontology concepts
    return spacy_terms, lexicon_output


#===================================================================================

if __name__ == "__main__":
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    print("Processing Life Biology Lexicon")
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    with open(bio_concepts_file, "r") as f:
        bio_concepts = set([t.strip() for t in f.readlines()])
        
    terms, lexicon = process_lexicon(lexicon, bio_concepts)
    
    write_spacy_docs(terms, terms_output_file)
    with open(lexicon_output_file, "w") as f:
        json.dump(lexicon, f, indent=4)
    
    print("Processing Life Biology Sentences")
    with open(life_input_file, "r") as f:
        life_bio_sentences = f.readlines()
        
    sentences_kb_spacy = []
    sentences_spacy = []
    for i, sent in enumerate(life_bio_sentences):
        if i % 500 == 0:
            print(f"Preprocessing life biology sentence {i}/{len(life_bio_sentences)}")
            
        # only add chapters 1-10 to subset used for kb matching
        spacy_sent = nlp(re.sub("^([\d*|summary]\.*)+\s*", "", sent))
        if int(sent.split(".")[1]) <= 10:
            sentences_kb_spacy.append(spacy_sent)
        sentences_spacy.append(spacy_sent)
        
    write_spacy_docs(sentences_spacy, life_output_file)
    write_spacy_docs(sentences_kb_spacy, life_kb_output_file)
    