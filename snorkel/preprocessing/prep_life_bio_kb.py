# Takes in preprocessed sentences and KB relations file 
# Creates separate sentence and key term files for each.
# For Life, does additional parsing to extract out the knowledge base lexicon and first 10 chapters.

# Author: Matthew Boggess
# Version: 4/22/20

# Data Source: 
#   - Outputs from Inquire knowledge base provided by Dr. Chaudhri

# Description: 
#   - Processes a dump from the Inquire knowledge base to produce the following output:
#       A Spacy NLP preprocessed set of biology terms extracted from the first 10 chapters
#       of Life Biology for the previous knowledge base

#===================================================================================

# Libraries

import json
import pickle
import spacy
from collections import defaultdict
import string
import re
import os
import itertools
from tqdm import tqdm
from spacy.tokens import DocBin
from data_processing_utils import read_spacy_docs

#===================================================================================
# Parameters

# Filepaths 
# directory holding KB dump of relations
relation_data_dir = "../../data/raw_data/life_bio"
# file mapping KB concepts to various text representations
lexicon_file = f"../../data/preprocessed_data/Life_Biology_kb_lexicon.json"
# file with full list of terms in Kb already spacy preprocessed 
kb_terms_file = f"../../data/preprocessed_data/Life_Biology_kb_key_terms_spacy"
output_data_dir = "../data"

# different groupings of relations whose KB dumps are stored in separate relation files 
relation_types = ['taxonomy', 'structure', 'process']

#===================================================================================
# Helper Functions

def extract_lemmas(lexicon, concept, instance):
    """ For a given concept, text representation in a relation this function 
    extracts all lemma representations of that concept and the associated text representations.
    """
    
    # pull out equivalent lemma, text representations from lexicon
    if concept in lexicon.keys():
        instance_texts = lexicon[concept]['text_representations']
        instance_lemmas = lexicon[concept]['lemma_representations']
    else:
        instance_texts = []
        instance_lemmas = []
        
    # add particular text from relation if not present
    if instance not in instance_texts:
        instance_texts.append(instance)
        instance_lemmas.append(' '.join([tok.lemma_ for tok in nlp(instance)]))
        
    # create a dictionary mapping all lemma forms to corresponding text forms for this concept
    instance_pairs = {}
    for lemma, text in zip(instance_lemmas, instance_texts):
        lemma = lemma.replace(' - ', ' ')
        if lemma in instance_pairs:
            instance_pairs[lemma].append(text)
        else:
            instance_pairs[lemma] = [text]
            
    return instance_pairs

def parse_relations(relations, relation_type, lexicon, relations_db):
    """ Parses all relations from a text file containing relations and adds them to
    the relations database.
    
    1. Parse each relation into concepts, relations, and text representation
    2. Extract all unique lemma representations and associated text representations 
       for each entity/event pair
    3. Add each unique relation pair to the relations database
    
    The relations_db is a dictionary mapping relation types (i.e. subclass-of) to a set of
    term pairs exhibiting that relation.
    """
    
    for r in tqdm(relations):
    
        if relation_type == 'taxonomy':
            c1, e1, relation, c2, e2 = [tok.strip() for tok in r.split('|')]
        else:
            _, _, c1, e1, relation, c2, e2 = [tok.strip() for tok in r.split('|')]

            c1 = c1.strip('_').replace('_ABOX_', '').rstrip(string.digits)
            c2 = c2.strip('_').replace('_ABOX_', '').rstrip(string.digits)
        
        if relation not in relations_db:
            relations_db[relation] = set() 
        
        # get all lemma representations for the 
        e1_pairs = extract_lemmas(lexicon, c1, e1)
        e2_pairs = extract_lemmas(lexicon, c2, e2)
        
        for pair in itertools.product(e1_pairs.keys(), e2_pairs.keys()):
            relations_db[relation].add(pair)
    
    return relations_db

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    # Load in KB lexicon and terms 
    with open(lexicon_file, 'r') as f:
        lexicon = json.load(f)
    relations_db = {}
    
    terms = DocBin().from_bytes(open(kb_terms_file, 'rb').read())
    terms = list(terms.get_docs(nlp.vocab))
    terms = set([(' '.join([t.lemma_ for t in term])).replace(' - ', ' ') for term in terms])
    
    # special handling for pulling out synonyms from the lexicon file
    print("Extracting Word-Pairs for Synonyms")
    relations_db['synonym'] = set()
    for concept in lexicon:
        lemmas = set([t.replace(' - ', ' ') for t in lexicon[concept]['lemma_representations']])
        for pair in itertools.product(lemmas, lemmas):
            if pair[0] != pair[1]:
                relations_db['synonym'].add(pair)
        
    # extract all word-pairs for relations
    print("Extracting Word-Pairs for Other Relations")
    for relation_type in relation_types:
        print(f"Parsing {relation_type} relations")
        with open(f"{relation_data_dir}/{relation_type}_relations.txt") as f:
            relations = f.readlines()
        relations_db = parse_relations(relations, relation_type, lexicon, relations_db)
    
    
    #print("Collecting Remaining Word-Pairs for No Relation")
    #for pair in tqdm(list(itertools.product(terms, terms))):
    #    
    #    # don't include self pairs
    #    if pair[0] == pair[1]:
    #        continue
    #    # don't include actual relations
    #    if pair in valid_pairs:
    #        continue
    #    # don't include reverse of valid relation pairs since those might get handled differently
    #    if (pair[1], pair[0]) in valid_pairs:
    #        print(pair)
    #        continue
    #    
    #    relations_db['no-relation'].add(pair)
    
    with open(f"{output_data_dir}/kb_bio101_relations_db.pkl", 'wb') as fid:
        pickle.dump(relations_db, fid)
        
    with open(f"{output_data_dir}/kb_bio101_terms.pkl", 'wb') as fid:
        pickle.dump(terms, fid)
    