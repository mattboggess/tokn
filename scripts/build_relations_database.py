import json
from io import StringIO
import spacy
from collections import defaultdict
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import warnings
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

def write_spacy_docs(docs, filepath):
    """ Writes serialized spacy docs to file.  
    
    Parameters
    ----------
    docs: list of spacy.tokens.doc.Doc
        List of spacy Docs to write to file 
    filepath: str
        File path to serialized spacy docs
    """
    doc_bin = DocBin()
    for doc in docs:
        doc_bin.add(doc)
        
    with open(filepath, "wb") as f:
        f.write(doc_bin.to_bytes())
    
def read_spacy_docs(filepath, nlp=None):
    """ Reads serialized spacy docs from a file into memory.
    
    Parameters
    ----------
    filepath: str
        File path to serialized spacy docs
    
    Returns
    -------
    list of spacy.tokens.doc.Doc
        List of spacy Docs loaded from file
    """
    
    if nlp is None:
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)
        
    with open(filepath, "rb") as f:
        data = f.read()
        
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs

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
    2. Extract all unique lemma representations and associated text representations for each entity/event pair
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

def get_closest_match(indices1, indices2):
    """
    Computes the closest pair of indices between two sets of indices.
    """
    
    if len(indices1) == 1 and len(indices2) == 1:
        return indices1[0], indices2[0]
    
    closest_match = (indices1[0], indices2[0])
    min_dist = np.abs(closest_match[0][0] - closest_match[1][0])
    for pair in itertools.product(indices1, indices2):
        dist = np.abs(pair[0][0] - pair[1][0])
        if dist < min_dist:
            closest_match = pair
            min_dist = dist
    
    return closest_match

def tag_bioes(tags, match_index, term_length):
    """ Updates tags for a text using the BIOES tagging scheme.

    B = beginning of term phrase
    I = interior of term phrase
    O = non-term
    E = end of term phrase
    S = singleton term

    Parameters
    ----------
    tags: list of str
        List of current BIOES tags for given tokenized text that we will be updating
    match_index: int
        Index at which the term was matched in the text
    term_length: int
        Number of tokens that compose the term ('cell wall' -> 2)

    Returns
    -------
    list of str
        Updated list of BIOES tags for the given tokenized text

    Examples
    --------

    >>> tag_bioes(['O', 'O', 'O', 'O'], 1, 2)
    ['O', 'B', 'E', 'O']

    >>> tag_bioes(['O', 'O', 'O', 'O'], 0, 1)
    ['S', 'O', 'O', 'O']

    >>> tag_bioes(['O', 'O', 'O', 'O'], 1, 3)
    ['O', 'B', 'I', 'E']

    """

    if term_length == 1:
        tags[match_index] = "S"
    else:
        for i in range(term_length):
            if i == 0:
                tags[match_index + i] = "B"
            elif i == term_length - 1:
                tags[match_index + i] = "E"
            else:
                tags[match_index + i] = "I"
    return tags

def tag_terms(text, terms, nlp=None):
    """ Tags all terms in a given span of text.
    """
    
    # default to Stanford NLP pipeline wrapped in Spacy
    if nlp is None:
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)
        
    # preprocess with spacy if needed
    if type(terms[0]) != spacy.tokens.doc.Doc:
        terms = [nlp(term) for term in terms]
    if type(text) != spacy.tokens.doc.Doc:
        text = nlp(text)

    normalized_text = [token.lemma_ for token in text]
    tokenized_text = [token.text for token in text]
    tagged_text = ['O'] * len(text)
    found_terms = defaultdict(lambda: {"text": [], "indices": [], "tag": []})
    
    # iterate through terms from longest to shortest
    terms = sorted(terms, key=len)[::-1]
    for term in terms:
        normalized_term = [token.lemma_ for token in term]

        for ix in range(len(text) - len(term)):

            if normalized_text[ix:ix + len(term)] == normalized_term:
                # only add term if not part of larger term
                if tagged_text[ix:ix + len(term)] == ["O"] * len(term):
                    lemma = " ".join(normalized_term)
                    found_terms[lemma]["text"].append(" ".join([token.text for token in term]))
                    found_terms[lemma]["indices"].append((ix, ix + len(term)))
                    found_terms[lemma]["tag"].append(" ".join([token.tag_ for token in term]))
                    tagged_text = tag_bioes(tagged_text, ix, len(term))
    
    return tokenized_text, tagged_text, found_terms

def insert_relation_tags(tokenized_text, indices):
    """ Inserts entity tags in a sentence denoting members of a relation. """
    
    # order tags by actual index in sentence
    indices = [i for ind in indices for i in ind]
    tags = ["<e1>", "</e1>", "<e2>", "</e2>"]
    order = np.argsort(indices)
    indices = [indices[i] for i in order]
    tags = [tags[i] for i in order]
    
    adjust = 0
    for ix, tag in zip(indices, tags):
        tokenized_text.insert(ix + adjust, tag)
        adjust += 1
    
    return tokenized_text

def add_relation(term_pair, term_info, tokenized_text, relations_db):
    """ For a given term pair, found at specific indices in a text, add the sentence to the
    relations database if it matches a relation(s) in the database, otherwise add it as a 
    no-relation instance.
    """
    tokenized_text = tokenized_text.copy()
    
    found_relation = False
    term_pair_key = " -> ".join(term_pair)
    
    # restrict to closest occurence of the two terms in the sentence
    indices = get_closest_match(term_info[term_pair[0]]["indices"], 
                                term_info[term_pair[1]]["indices"])
    
    # tag term pair in the sentence
    tokenized_text = insert_relation_tags(tokenized_text, indices)
    
    for relation in relations_db:
        if term_pair_key in relations_db[relation]: 
            
            # add sentence to relations database 
            found_relation = True
            relations_db[relation][term_pair_key]["sentences"].append(tokenized_text)
            
    if not found_relation:

        if term_pair_key in relations_db["no-relation"]:
            relations_db["no-relation"][term_pair_key]["sentences"].append(tokenized_text)
        else:
            relations_db["no-relation"][term_pair_key] = {
                "sentences": [tokenized_text], 
                "e1_representations": term_info[term_pair[0]]["text"],
                "e2_representations": term_info[term_pair[1]]["text"]}
      
    return relations_db

def tag_relations(text, terms, relations_db, nlp=None):
    
    # default to Stanford NLP pipeline wrapped in Spacy
    if nlp is None:
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)
        
    # preprocess with spacy if needed
    if type(terms[0]) != spacy.tokens.doc.Doc:
        terms = [nlp(term) for term in terms]
    if type(text) != spacy.tokens.doc.Doc:
        text = nlp(text)

    normalized_text = [token.lemma_ for token in text]
    tokenized_text = [token.text for token in text]
    
    tokenized_text, tagged_text, found_terms_info = tag_terms(text, terms, nlp)

    found_terms = list(found_terms_info.keys())
    for i in range(len(found_terms) - 1):
        for j in range(i + 1, len(found_terms)):
            term_pair = (found_terms[i], found_terms[j])
            relations_db = add_relation(term_pair, found_terms_info, tokenized_text, 
                                        relations_db)
            term_pair_reverse = (found_terms[i], found_terms[j])
            relations_db = add_relation(term_pair_reverse, found_terms_info, tokenized_text, 
                                        relations_db)
    
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
        with open(lexicon_file, "r") as f:
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
            print("Parsing {relation_type} relations")
            file = f"{raw_data_dir}/{relation_type}_relations.txt"
            with open(file) as f:
                relations = f.readlines()
            relations_db = parse_relations(relations, relation_type, lexicon, relations_db, 
                                           include_relations)
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
