##########
# Takes in preprocessed sentences and KB relations file 
# Creates separate sentence and key term files for each.
# For Life, does additional parsing to extract out the knowledge base lexicon and first 10 chapters.
##########
import json
import spacy
from collections import defaultdict
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import warnings
import string
import pandas as pd
import re
import os
import numpy as np
import itertools
from spacy.tokens import DocBin
from data_processing_constants import INCLUDE_RELATIONS
from data_processing_utils import tag_relations, read_spacy_docs

def extract_lemmas(lexicon, concept, instance):
    """ For a given concept, text representation in a relation this function 
    extracts all lemma representations of that concept and the associated text representations.
    """
    
    # pull out equivalent lemma, text representations from lexicon
    if concept in lexicon.keys():
        instance_texts = lexicon[concept]["text_representations"]
        instance_lemmas = lexicon[concept]["lemma_representations"]
    else:
        instance_texts = []
        instance_lemmas = []
        
    # add particular text from relation if not present
    if instance not in instance_texts:
        instance_texts.append(instance)
        instance_lemmas.append(" ".join([tok.lemma_ for tok in nlp(instance)]))
        
    # create a dictionary mapping all lemma forms to corresponding text forms for this concept
    instance_pairs = {}
    for lemma, text in zip(instance_lemmas, instance_texts):
        if lemma in instance_pairs:
            instance_pairs[lemma].append(text)
        else:
            instance_pairs[lemma] = [text]
            
    return instance_pairs

def parse_relations(relations, relation_type, lexicon, relations_db, include_relations=None):
    """ Parses all relations from a text file containing relations and adds them to
    the relations database.
    
    1. Parse each relation into concepts, relations, and text representation
    2. Extract all unique lemma representations and associated text representations 
       for each entity/event pair
    3. Add each unique relation pair to the relations database
    """
    
    for i, r in enumerate(relations):
    
        if i % 500 == 0:
            print(f"Parsing relation {i}/{len(relations)}")
    
        if relation_type == "taxonomy":
            c1, e1, relation, c2, e2 = [tok.strip() for tok in r.split("|")]
        elif relation_type == "structure" or relation_type == "process":
            _, _, c1, e1, relation, c2, e2 = [tok.strip() for tok in r.split("|")]

            c1 = c1.strip("_").replace("_ABOX_", "").rstrip(string.digits)
            c2 = c2.strip("_").replace("_ABOX_", "").rstrip(string.digits)
        
        if include_relations and relation not in include_relations:
            continue
    
        if relation not in relations_db:
            relations_db[relation] = {}
        
        e1_pairs = extract_lemmas(lexicon, c1, e1)
        e2_pairs = extract_lemmas(lexicon, c2, e2)
        
        for pair in itertools.product(e1_pairs.keys(), e2_pairs.keys()):
            relations_db[relation][" -> ".join(pair)] = {"sentences": [], 
                                                         "concept_pair": " -> ".join((c1, c2)),
                                                         "e1_representations": e1_pairs[pair[0]], 
                                                         "e2_representations": e2_pairs[pair[1]]}
    
    return relations_db

if __name__ == "__main__":
    
    # set up data folders
    raw_data_dir = "../data/raw_data/life_bio"
    output_data_dir = "../data/relation_extraction"
    preprocessed_data_dir = "../data/preprocessed_data"
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    # Load in KB lexicon and terms 
    lexicon_file = f"{preprocessed_data_dir}/Life_Biology_kb_lexicon.json"
    kb_terms_file = f"{preprocessed_data_dir}/Life_Biology_kb_key_terms_spacy"
    with open(lexicon_file, "r") as f:
        lexicon = json.load(f)
    terms = read_spacy_docs(kb_terms_file, nlp)
        
    # extract all word-pairs for relations
    print("Extracting Word-Pairs for Relations")
    relations_db = {"no-relation": {}}
    for relation_type in ["taxonomy", "structure", "process"]:
        print(f"Parsing {relation_type} relations")
        file = f"{raw_data_dir}/{relation_type}_relations.txt"
        with open(file) as f:
            relations = f.readlines()
        relations_db = parse_relations(relations, relation_type, lexicon, relations_db, 
                                       INCLUDE_RELATIONS)
    
    # load biology textbook sentences 
    bio_textbooks = ["Life_Biology", "Biology_2e"]
    sentences = []
    for textbook in bio_textbooks:
        sentences += read_spacy_docs(f"{preprocessed_data_dir}/{textbook}_sentences_spacy", nlp)
        
    # tag sentences with relations and add to relations database
    print("Adding tagged sentences to relations database")
    relations_db_output_file = f"{output_data_dir}/relations_db.json"
    for i, sentence in enumerate(sentences):
        if i % 500 == 0:
            print(f"Tagging relations for sentence {i}/{len(sentences)}")
        relations_db = tag_relations(sentence, terms, relations_db, nlp)
    with open(relations_db_output_file, "w") as f:
        json.dump(relations_db, f, indent=4)
