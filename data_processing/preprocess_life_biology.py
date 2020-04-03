# Preprocesses Life Biology Chs. 1-10, 39-52 sentences using Stanford NLP pipeline.
# Takes a dump of the Inquire knowledge base's lexicon and processes it to get text representations
# and entity/event labels for each biology concept in the knowledge base.

# Author: Matthew Boggess
# Version: 4/3/20

# Data Source: 
#   - Dump of individual sentences of Life Biology chapters provided by Dr. Chaudhri
#   - Outputs from Inquire knowledge base provided by Dr. Chaudhri

# Description: 
#   - Runs individual sentences from chapters 1-10 and 39-52 of life biology through Stanford
#     NLP pipeline including tokenization, pos tagging, lemmatization, etc. Saves ch. 1-10 on its
#     own in order to restrict to knowledge base chapters as well as saves all together.
#   - Processes a dump from the Inquire knowledge base to produce the following two outputs:
#       1. A 'lexicon' in json format mapping each biology concept in the kb to a list of text,
#          representations, the lemmatized form of those representations, and event/entity label
#       2. A Stanford NLP preprocessed set of biology terms that can be used to tag the life
#          biology sentences for term extraction

#===================================================================================

# Libraries

import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
from io import StringIO
import pandas as pd
import os
import warnings
import re
from tqdm import tqdm
import json

#===================================================================================

# Parameters

## Filepaths

# data directories
raw_data_dir = "../data/raw_data/life_bio"
preprocessed_data_dir = "../data/preprocessed_data"
if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)

# lexicon/kb input and output files for lexicon 
lexicon_input_file = f"{raw_data_dir}/kb_lexicon.txt"
lexicon_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_lexicon.json"
bio_concepts_file = f"{raw_data_dir}/kb_biology_concepts.txt"
terms_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_key_terms_spacy"
terms_output_vocab_file = f"{preprocessed_data_dir}/Life_Biology_kb_key_terms_spacy_vocab"

# sentence input and output files for life biology
input_sentences_file = f"{raw_data_dir}/life_bio_selected_sentences.txt"
kb_output_sentences_file = f"{preprocessed_data_dir}/Life_Biology_kb_sentences_spacy"
output_sentences_file = f"{preprocessed_data_dir}/Life_Biology_sentences_spacy"
output_sentences_vocab_file = f"{preprocessed_data_dir}/Life_Biology_sentences_spacy_vocab"

## Important Enumerations 

# text representations of concepts that are too general and thus problematic for text matching
exclude_representations = ['object', 'aggregate', 'group', 'thing', 'region']

#===================================================================================

# Helper Functions

def process_lexicon(lexicon, bio_concepts):
    """ 
    Processes lexicon information from the Inquire knowledge base that provides information
    about how each biology concept in the knowledge base is represented in actual text. Specifically
    it produces a json file mapping each concept to a list of text representations, their lemma
    forms, and a entity/event label. Additionally, this function aggregates all text representations
    across all terms into a single list after running each through the Stanford NLP pipeline.
    """
    
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, 
                          names=['concept', 'relation', 'text', 'pos'])
    
    concept_types = lexicon.query("text in ['Entity', 'Event']")
    lexicon = lexicon.query("text not in ['Entity', 'Event']")

    # create mapping from kb concept to unique text representations excluding text
    # representations are too general (i.e. 'object')
    lexicon = lexicon[~lexicon.text.str.contains('Concept-Word-Frame')]
    lexicon = lexicon.groupby('concept')['text'].apply(
        lambda x: list(set([t for t in x if t not in exclude_representations]))).reset_index()
    
    # filter out too general upper ontology words, relation concepts, and 
    # concepts that only have representations that are too general 
    lexicon = lexicon[lexicon.concept.isin(bio_concepts)]
    lexicon = lexicon[lexicon.text.map(len) > 0]
    lexicon = lexicon[lexicon.text.apply(lambda x: 'Relation' not in x)]

    # spacy process terms to get lemmas
    spacy_terms = []
    lexicon_output = {}
    print("Running text representations for each concept through Stanford NLP pipeline")
    for concept in tqdm(lexicon.concept):
        
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

    return spacy_terms, lexicon_output

#===================================================================================

if __name__ == "__main__":
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings("ignore")
    
    print("Processing Life Biology Lexicon")
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    with open(bio_concepts_file, "r") as f:
        bio_concepts = set([t.strip() for t in f.readlines()])
        
    terms, lexicon = process_lexicon(lexicon, bio_concepts)
    
    write_spacy_docs(terms, nlp.vocab, terms_output_file, terms_output_vocab_file)	
    with open(lexicon_output_file, "w") as f:
        json.dump(lexicon, f, indent=4)
    
    print("Processing Life Biology Sentences")
    with open(input_sentences_file, "r") as f:
        life_bio_sentences = f.readlines()
        
    sentences_kb_spacy = []
    sentences_spacy = []
    print("Running chapter sentences through Stanford NLP pipeline")
    for sent in tqdm(life_bio_sentences):
            
        # only add chapters 1-10 to output restricted to knowledge base sentences
        spacy_sent = nlp(re.sub("^([\d*|summary]\.*)+\s*", "", sent))
        if int(sent.split(".")[1]) <= 10:
            sentences_kb_spacy.append(spacy_sent)
        sentences_spacy.append(spacy_sent)
        
    write_spacy_docs(sentences_spacy, nlp.vocab, output_sentences_file, output_sentences_vocab_file)	
    write_spacy_docs(sentences_kb_spacy, nlp.vocab, kb_output_sentences_file, output_sentences_vocab_file)	
    
