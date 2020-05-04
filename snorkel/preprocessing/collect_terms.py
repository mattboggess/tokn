# Preprocesses Life Biology Chs. 1-10, 39-52 sentences using Spacy NLP pipeline.
# Takes a dump of the Inquire knowledge base's lexicon and processes it to get text representations
# and entity/event labels for each biology concept in the knowledge base.

# Author: Matthew Boggess
# Version: 4/21/20

# Data Source: 
#   - Outputs from Inquire knowledge base provided by Dr. Chaudhri

# Description: 
#   - Processes a dump from the Inquire knowledge base to produce the following output:
#       A Spacy NLP preprocessed set of biology terms extracted from the first 10 chapters
#       of Life Biology for the previous knowledge base

#===================================================================================

# Libraries

import spacy
from data_processing_utils import write_spacy_docs
from io import StringIO
import pandas as pd
import os
import re
from tqdm import tqdm
import json

#===================================================================================

# Parameters

## Filepaths

# data directories
life_bio_data_dir = "../../data/raw_data/life_bio"
openstax_data_dir = "../../data/raw_data/openstax/openstax_provided_book_import/sentence_files"
preprocessed_data_dir = "../data"
if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)
    
bio_concepts_file = f"{life_bio_data_dir}/kb_biology_concepts.txt"
lexicon_input_file = f"{life_bio_data_dir}/kb_lexicon.txt"
terms_output_file = f"{preprocessed_data_dir}/biology_terms_spacy"
terms_vocab_output_file = f"{preprocessed_data_dir}/biology_terms_spacy_vocab"

# text representations of concepts that are too general and thus problematic for text matching
exclude_terms = ['object', 'aggregate', 'group', 'thing', 'region', 'center', 'response',
                 'series', 'unit', 'result', 'normal', 'divide', 'whole', 'someone', 'somebody']

#===================================================================================

# Helper Functions

def process_lexicon(lexicon, bio_concepts):
    """ 
    Processes lexicon information from the Inquire knowledge base that provides information
    about how each biology concept in the knowledge base is represented in actual text. Specifically
    it produces a json file mapping each concept to a list of text representations, their lemma
    forms, and a entity/event label. Additionally, this function aggregates all text representations
    across all terms into a single list after running each through the Spacy NLP pipeline.
    """
    
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, engine='python',
                          names=['concept', 'relation', 'text', 'pos'])
    
    concept_types = lexicon.query("text in ['Entity', 'Event']")
    lexicon = lexicon.query("text not in ['Entity', 'Event']")

    # create mapping from kb concept to unique text representations excluding text
    # representations are too general (i.e. 'object')
    lexicon = lexicon[~lexicon.text.str.contains('Concept-Word-Frame')]
    lexicon = lexicon.groupby('concept')['text'].apply(
        lambda x: list(set([t for t in x if t not in exclude_terms]))).reset_index()
    
    # filter out too general upper ontology words, relation concepts, and 
    # concepts that only have representations that are too general 
    lexicon = lexicon[lexicon.concept.isin(bio_concepts)]
    lexicon = lexicon[lexicon.text.map(len) > 0]
    lexicon = lexicon[lexicon.text.apply(lambda x: 'Relation' not in x)]

    # spacy process terms to get lemmas
    spacy_terms = []
    lexicon_output = {}
    print("Running text representations for each concept through Spacy NLP pipeline")
    for concept in tqdm(lexicon.concept):
        
        # extract text representations for the concept
        terms = list(lexicon.loc[lexicon.concept == concept, 'text'])[0]
        terms = [t.replace('"', '').strip().replace('-', ' ') for t in terms]
        
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

if __name__ == '__main__':
    
    seen_lemmas = set()
    terms = []
    
    # initialize Spacy NLP pipeline
    nlp = spacy.load('en_core_web_sm')
    
    print("Extracting Life Biology KB terms")
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    with open(bio_concepts_file, "r") as f:
        bio_concepts = set([t.strip() for t in f.readlines()])
    life_terms, _ = process_lexicon(lexicon, bio_concepts)
    
    for term in life_terms:
        term_lemma = ' '.join(t.lemma_ for t in term)
        if term_lemma in seen_lemmas:
            continue
        seen_lemmas.add(term_lemma)
        terms.append(term)
    
    print("Extracting OpenStax Biology terms")
    textbook_data = pd.read_csv(f"{openstax_data_dir}/sentences_Biology_2e_parsed.csv")
    key_terms = textbook_data[textbook_data.section_name == "Key Terms"].sentence
    openstax_terms = []
    for key_term in tqdm(key_terms):
        kts = parse_openstax_terms(key_term)
        if len(kts):
            openstax_terms += [nlp(kt) for kt in kts]
    
    for term in openstax_terms:
        term_lemma = ' '.join(t.lemma_ for t in term)
        if term_lemma in seen_lemmas:
            continue
        seen_lemmas.add(term_lemma)
        terms.append(term)
        
    print(f"Extracted {len(terms)} biology terms")
        
    # remove too common/problematic 
    exclude_lemmas = [' '.join([x.lemma_ for x in nlp(term)]) for term in exclude_terms]
    filtered_terms = []
    for term in terms:
        term_lemma = ' '.join(x.lemma_ for x in term)
        if term_lemma in exclude_lemmas:
            continue
        filtered_terms.append(term)
    terms = filtered_terms
    print(f"{len(terms)} biology terms remaining for tagging after removing too common terms")
    
    write_spacy_docs(filtered_terms, nlp.vocab, terms_output_file, terms_vocab_output_file)	
