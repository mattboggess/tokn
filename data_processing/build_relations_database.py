import json
from io import StringIO
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

MERONYM_RELATIONS = ["has-part", "has-region", "element", "possesses", "material"]
SPATIAL_RELATIONS = ["is-at", "is-inside", "is-outside", "abuts", "between"]
TAXONOMY_RELATIONS = ["subclass-of", "instance-of"]
INCLUDE_RELATIONS = MERONYM_RELATIONS + SPATIAL_RELATIONS + TAXONOMY_RELATIONS 


def process_lexicon(lexicon):
    """ Takes in a lexicon consisting of concept text representation pairs and turns this into a 
    list of Spacy processed terms and a lexicon csv mapping KB concepts to lists of text 
    representations and their lemma forms.
    """
    
    # get rid of extra column and read in as dataframe
    lexicon = lexicon.replace(' | "n"', '').replace('"', '')
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, 
                          names=["concept", "relation", "text"])

    # create mapping from kb concept to unique text representations
    lexicon = lexicon[~lexicon.text.str.contains("Concept-Word-Frame")]
    lexicon = lexicon.groupby("concept")["text"].apply(lambda x: list(set(x))).reset_index()

    # spacy process terms to get lemmas
    spacy_terms = []
    lemmas = []
    for concept in lexicon.concept:
        terms = list(lexicon.loc[lexicon.concept == concept, "text"])[0]
        spacy_terms_tmp = [nlp(term) for term in terms]
        lemma_terms = list([" ".join([tok.lemma_ for tok in t]) for t in spacy_terms_tmp])
        spacy_terms += spacy_terms_tmp
        lemmas.append(lemma_terms)

    lexicon["lemmas"] = lemmas
    return spacy_terms, lexicon

def extract_lemmas(lexicon, concept, instance):
    """ For a given concept, text representation in a relation this function 
    extracts all lemma representations of that concept and the associated text representations.
    """
    
    # pull out equivalent lemma, text representations from lexicon
    if concept in lexicon.concept:
        instance_texts = lexicon.at[lexicon.concept == concept, "text"]
        instance_lemmas = lexicon.at[lexicon.concept == concept, "lemma"]
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
        elif relation_type == "structure":
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
                                                         "e1_representations": e1_pairs[pair[0]], 
                                                         "e2_representations": e2_pairs[pair[1]]}
    
    return relations_db


if __name__ == "__main__":
    
    # set up data folders
    raw_data_dir = "../data/relation_extraction/raw_data"
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    output_data_dir = "../data/relation_extraction/processed_data"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    # aggregate text & lemma representations of KB concepts
    print("Collecting Lexicon Representations")
    lexicon_input_file = f"{raw_data_dir}/kb_lexicon.txt"
    lexicon_output_file = f"{output_data_dir}/lexicon.csv"
    terms_output_file = f"{output_data_dir}/kb_terms_spacy"
    
    if not os.path.exists(lexicon_output_file) or not os.path.exists(terms_output_file):
        with open(lexicon_input_file, "r") as f:
            lexicon = f.read()
        terms, lexicon = process_lexicon(lexicon)
        write_spacy_docs(terms, terms_output_file)
        lexicon.to_csv(lexicon_output_file, index=False)
    else:
        lexicon = pd.read_csv(lexicon_output_file)
        terms = read_spacy_docs(terms_output_file, nlp)
        
    # extract all word-pairs for relations
    print("Extracting Word-Pairs for Relations")
    intermed_relations_db_file = f"{output_data_dir}/relations_db_intermediate.json"
    if os.path.exists(intermed_relations_db_file):
        with open(intermed_relations_db_file, "r") as f:
            relations_db = json.load(f)
    else:
        relations_db = {"no-relation": {}}
        for relation_type in ["taxonomy", "structure"]:
            print(f"Parsing {relation_type} relations")
            file = f"{raw_data_dir}/{relation_type}_relations.txt"
            with open(file) as f:
                relations = f.readlines()
            relations_db = parse_relations(relations, relation_type, lexicon, relations_db, 
                                           INCLUDE_RELATIONS)
        with open(intermed_relations_db_file, "w") as f:
            json.dump(relations_db, f, indent=4)
    
    # clean and preprocess biology textbook sentences
    print("Preprocessing sentences")
    # openstax bio
    print("Preprocessing openstax bio...")
    openstax_input_file = f"{raw_data_dir}/final_bio_parsed.csv"
    openstax_output_file = f"{output_data_dir}/openstax_biology_sentences_spacy"
    if os.path.exists(openstax_output_file):
        stax_sentences = read_spacy_docs(openstax_output_file, nlp)
    else:
        stax_bio_sentences = pd.read_csv(openstax_input_file)
        exclude_sections = ["Preface", "Chapter Outline", "Index", "Chapter Outline", 
                            "Critical Thinking Questions", "Visual Connection Questions", 
                            "Key Terms", "Review Questions", 
                            "The Periodic Table of Elements", "Measurements and the Metric System"]
        stax_bio_sentences = stax_bio_sentences[~(stax_bio_sentences.section_name.isin(exclude_sections))]
        stax_sentences = []
        for i, sent in enumerate(stax_bio_sentences.sentence):
            if i % 500 == 0:
                print(f"Preprocessing openstax sentence {i}/{len(stax_bio_sentences.sentence)}")
            stax_sentences.append(nlp(sent))
        write_spacy_docs(stax_sentences, openstax_output_file)
        
    # life bio
    print("Preprocessing life bio...")
    life_input_file = f"{raw_data_dir}/life_bio_selected_sentences.txt"
    life_output_file = f"{output_data_dir}/life_biology_sentences_spacy"
    if os.path.exists(life_output_file):
        life_sentences = read_spacy_docs(life_output_file, nlp)
    else:
        with open(life_input_file, "r") as f:
            life_bio_sentences = f.readlines()
        life_sentences = []
        for i, sent in enumerate(life_bio_sentences):
            if i % 500 == 0:
                print(f"Preprocessing life sentence {i}/{len(life_bio_sentences)}")
            life_sentences.append(nlp(re.sub("^(\d*\.*)+\s*", "", sent)))
        write_spacy_docs(life_sentences, life_output_file)
    
    sentences = stax_sentences + life_sentences
    
    # tag sentences with relations and add to relations database
    print("Adding tagged sentences to relations database")
    relations_db_output_file = f"{output_data_dir}/relations_db.json"
    for i, sentence in enumerate(sentences):
        if i % 500 == 0:
            print(f"Tagging relations for sentence {i}/{len(sentences)}")
        relations_db = tag_relations(sentence, terms, relations_db, nlp)
    with open(relations_db_output_file, "w") as f:
        json.dump(relations_db, f, indent=4)
