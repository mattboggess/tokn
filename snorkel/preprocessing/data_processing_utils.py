from spacy.tokens import DocBin
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import spacy
import numpy as np
import itertools
from collections import defaultdict


def tag_relations(text, terms, relations_db, nlp=None):
    """ Tags all terms in a given text and then extracts all relations between pairs of these terms.
    
    Parameters
    ----------
    text: str 
        Input text that we want to add relations for 
    terms: list of str | spacy.tokens.doc.Doc
        List of terms which we will tag the sentence and look for relations between 
    relations_db: dict
        Dictionary mapping relations to word pairs and sentences we will update
    nlp: Spacy nlp pipeline
        Optional Spacy nlp pipeline object used for processing text
    
    Returns
    -------
    dict
        Updated relations dictionary database
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

    result = tag_terms(text, terms, nlp)
    tokenized_text = result["tokenized_text"]
    tagged_text = result["tags"]
    found_terms_info = result["found_terms"]

    found_terms = list(found_terms_info.keys())
    for i in range(len(found_terms) - 1):
        for j in range(i + 1, len(found_terms)):
            term_pair = (found_terms[i], found_terms[j])
            relations_db = add_relation(term_pair, found_terms_info, tokenized_text, 
                                        relations_db)
            term_pair_reverse = (found_terms[j], found_terms[i])
            relations_db = add_relation(term_pair_reverse, found_terms_info, tokenized_text, 
                                        relations_db)
    
    return relations_db

def add_relation(term_pair, term_info, tokenized_text, relations_db):
    """ For a given term pair, found at specific indices in a text, add the text to the
    relations database if it matches a relation(s) in the database, otherwise add it as a 
    no-relation instance.
    
    Parameters
    ----------
    term_pair: tuple of str
        Pair of terms for which we want to add relations for
    term_info: dict
        Maps lemmatized form of terms to indices and term info that were found in tokenized text
    tokenized_text: list of str
        List of tokenized text for the given sentence we want to tag relations
    relations_db: dict
        Dictionary mapping relations to word pairs and sentences we will update
    
    Returns
    -------
    dict
        Updated relations dictionary database
    """
    tokenized_text = tokenized_text.copy()
    
    found_relation = False
    term_pair_key = " -> ".join(term_pair)
    
    # restrict to closest occurence of the two terms in the sentence
    indices = get_closest_match(term_info[term_pair[0]]["indices"], 
                                term_info[term_pair[1]]["indices"])
    
    term1_text = " ".join(tokenized_text[indices[0][0]:indices[0][1]])
    term2_text = " ".join(tokenized_text[indices[1][0]:indices[1][1]])
    
    # tag term pair in the sentence
    tokenized_text = " ".join(insert_relation_tags(tokenized_text, indices))
    
    for relation in relations_db:
        if term_pair_key in relations_db[relation]: 
            
            # add sentence to relations database 
            found_relation = True
            relations_db[relation][term_pair_key]["sentences"].append(tokenized_text)
            if term1_text not in relations_db[relation][term_pair_key]["e1_representations"]:
                relations_db[relation][term_pair_key]["e1_representations"].append(term1_text)
            if term2_text not in relations_db[relation][term_pair_key]["e2_representations"]:
                relations_db[relation][term_pair_key]["e2_representations"].append(term2_text)
            
    if not found_relation:

        if term_pair_key in relations_db["no-relation"]:
            relations_db["no-relation"][term_pair_key]["sentences"].append(tokenized_text)
            if term1_text not in relations_db["no-relation"][term_pair_key]["e1_representations"]:
                relations_db["no-relation"][term_pair_key]["e1_representations"].append(term1_text)
            if term2_text not in relations_db["no-relation"][term_pair_key]["e2_representations"]:
                relations_db["no-relation"][term_pair_key]["e2_representations"].append(term2_text)
        else:
            relations_db["no-relation"][term_pair_key] = {
                "sentences": [tokenized_text], 
                "e1_representations": [term1_text],
                "e2_representations": [term2_text]}
      
    return relations_db


def insert_relation_tags(tokenized_text, indices):
    """ Inserts entity tags in a sentence denoting members of a relation. 
    
    The first entity in the relation is tagged with <e1> </e1> and second with <e2> </e2>
    
    Parameters
    ----------
    tokenized_text: list of str
        Spacy tokenized text list
    indices: ((int, int), (int, int))
        Pairs of indices denoting where the two entities in the sentences are to be tagged
    
    Returns
    -------
    list of str
        Modified tokenized text list with entity tags added
    
    Examples
    --------
    
    >>> insert_relation_tags(["A", "biologist", "will", "tell", "you", "that", "a", "cell", 
                              "contains", "a", "cell", "wall", "."], ((1, 2), (10, 12)))
    ["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", "a", "cell", 
     "contains", "a", "<e2>", "cell", "wall", "</e2>", "."]
    """
    
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

def tag_terms(text, terms, nlp=None):
    """ Identifies and tags any terms in a given input text.

    Searches through the input text and finds all terms (single words and phrases) that are present
    in the list of provided terms. Returns a list of found terms with indices and POS tagging as 
    well as a BIOES tagged version of the sentence denoting where the terms are in the sentences. 
    
    Additionally classifies terms as either entities or events and annotates the presence of the
    terms in the original sentence with these labels.
  
    Uses spacy functionality to tokenize and lemmatize for matching text (it is recommended to 
    preprocess by Spacy before inputting to prevent repeated work if calling multiple times).

    Gives precedence to longer terms first so that terms that are part of a larger term phrase
    are ignored (i.e. match 'cell wall', not 'cell' within the phrase cell wall). 

    Parameters
    ----------
    text: str | spacy.tokens.doc.Doc
        Input text that will be/are preprocessed using spacy and searched for terms
    terms: list of str | list of spacy.tokens.doc.Doc
        List of input terms that will be/are preprocessed using spacy. 
    nlp: 
        Spacy nlp pipeline object that will tokenize, POS tag, lemmatize, etc. 

    Returns
    -------
    dict with four entries: 
        tokenized_text: tokenized text as list of tokens
        tags: list of BIOES tags for the tokenized text
        annotated_text: original text with <entity> and <event> tags put around found terms 
        found_terms: list of found terms each with list of indices where matches were found,
        basic part of speech information, and entity/event tag

    Examples
    --------
    >>> tag_text('A biologist will tell you that a cell contains a cell wall.', 
                 ['cell', 'cell wall', 'biologist'])
    
    {'tokenized_text': ['A', 'biologist', 'will', 'tell', 'you', 'that', 'a', 'cell', 'contains', 
                        'a', 'cell', 'wall', '.'], 
     'tags': ['O', 'S', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'B', 'E', 'O'], 
     'annotated_text': 'A <entity>biologist</entity> will tell you that a <entity>cell</entity> 
                        contains a <entity>cell wall</entity>.', 
     'found_terms': {
         'cell wall': {'text': ['cell wall'], 'indices': [(10, 12)], 'pos': ['NN NN'], 
                       'type': ['entity']}, 
         'biologist': {'text': ['biologist'], 'indices': [(1, 2)], 'pos': ['NN'], 
                       'type': ['entity']}, 
         'cell': {'text': ['cell'], 'indices': [(7, 8)], 'pos': ['NN'], 'type': ['entity']}}}
    """
    from spacy.lang.en.stop_words import STOP_WORDS
    spacy.tokens.token.Token.set_extension('workaround', default='', force=True)
    
    HEURISTIC_TOKENS = ['-', 'plant', 'substance', 'atom']
    HEURISTIC_MAPPING = {
        'alpha': 'α',
        'beta': 'β',
        'prime': '′'
    }
    
    # default to Stanford NLP pipeline wrapped in Spacy
    if nlp is None:
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)
        
    # preprocess with spacy if needed
    if type(terms[0]) != spacy.tokens.doc.Doc:
        terms = [nlp(term) for term in terms]
    if type(text) != spacy.tokens.doc.Doc:
        text = nlp(text)
    
    # set up a custom representation of the text where we can add term type annotations
    for token in text:
        token._.workaround = token.text_with_ws

    lemmatized_text = [token.lemma_ for token in text]
    tokenized_text = [token.text for token in text]
    tags = ['O'] * len(text)
    found_terms = defaultdict(lambda: {'text': [], 'indices': [], 'pos': [], 'type': []})
    
    # iterate through terms from longest to shortest to ensure we tag the largest possible phrase
    terms = sorted(terms, key=len)[::-1]
    for spacy_term in terms:
        term_length = len(spacy_term)
        lemma_term_list = [token.lemma_ for token in spacy_term]
        text_term_list = [token.text for token in spacy_term]
        term_lemma = ' '.join(lemma_term_list)
        term_text = ' '.join(text_term_list).lower()
        
        # skip stop words
        if term_text in STOP_WORDS or term_lemma in STOP_WORDS:
            continue
        
        # additional check to check for simple plural of uncommon biology terms
        match_uncommon_plural = lemma_term_list.copy()
        match_uncommon_plural[-1] = match_uncommon_plural[-1] + 's'

        # additional check using dropped heuristics on lemmatized version
        match_heuristic = []
        if lemma_term_list[0] not in HEURISTIC_TOKENS:
            for token in lemma_term_list:
                if token not in HEURISTIC_TOKENS:
                    match_heuristic += token.split('-')
            heuristic_length = len(match_heuristic)
        else:
            heuristic_term = lemma_term_list
            heuristic_length = len(lemma_term_list)
        
        # additional check replacing KB speak with appropriate text symbols
        match_replace = []
        for token in lemma_term_list:
            if token in HEURISTIC_MAPPING:
                match_replace.append(HEURISTIC_MAPPING[token])
            else:
                match_replace.append(token)
        
        for ix in range(len(text) - term_length):
            
            heuristic_match = (lemmatized_text[ix:ix + heuristic_length] == match_heuristic)
            plural_match = (lemmatized_text[ix:ix + term_length] == match_uncommon_plural)
            lemma_match = (lemmatized_text[ix:ix + term_length] == lemma_term_list)
            replace_match = (lemmatized_text[ix:ix + term_length] == match_replace)
            
            if len(term_lemma) <= 2:
                text_match = ([t for t in tokenized_text[ix:ix + term_length]] == \
                              [t for t in text_term_list])
            else:
                text_match = ([t.lower() for t in tokenized_text[ix:ix + term_length]] == \
                              [t.lower() for t in text_term_list])
                
            valid_match = heuristic_match or plural_match or text_match or lemma_match or replace_match
            
            if valid_match:
                
                if heuristic_match and not lemma_match:
                    match_length = heuristic_length
                else:
                    match_length = term_length
                
                term_text = ' '.join([t.text for t in text[ix:ix + match_length]])
                term_tag = ' '.join([t.tag_ for t in text[ix:ix + match_length]])
                
                # only tag term if not part of larger term
                if tags[ix:ix + match_length] == ['O'] * match_length:
                    
                    # classify term type
                    term_type = determine_term_type(spacy_term)
                    
                    # collect term information
                    found_terms[term_lemma]['text'].append(term_text)
                    found_terms[term_lemma]['indices'].append((ix, ix + match_length))
                    found_terms[term_lemma]['pos'].append(term_tag)
                    found_terms[term_lemma]['type'].append(term_type)
                    
                    # update sentence tags
                    tags = tag_bioes(tags, ix, match_length)
                    
                    # annotate token representations with term type
                    text[ix]._.workaround = f'<{term_type}>' + text[ix]._.workaround
                    end_ix = ix + match_length - 1
                    if text[end_ix]._.workaround.endswith(' '):
                        text[end_ix]._.workaround = text[end_ix]._.workaround[:-1] + f'</{term_type}> '
                    else:
                        text[end_ix]._.workaround += f'</{term_type}>'
                    
    # reconstruct fully annotated input text
    annotated_text = ''
    for token in text:
        annotated_text += token._.workaround
    
    return {
        'tokenized_text': tokenized_text, 
        'tags': tags, 
        'annotated_text': annotated_text,
        'found_terms': dict(found_terms)
    }

def determine_term_type(term):
    """ Categorizes a term as either entity or event based on several heuristics.

    Parameters
    ----------
    term: spacy.tokens.doc.Doc
        Spacy preprocessed representation of the term 

    Returns
    -------
    str ('entity' | 'event')
        The class of the term
    """
    
    nominals = ["ation", "ition", "ption", "ing", "sis", "lism", "ment", "sion"]
    event_keywords = ["process", "cycle"]
    
    # key words that indicate events despite being nouns 
    if any([ek in term.text.lower() for ek in event_keywords]):
        term_type = "event"
    # key endings indicating a nominalized form of an event 
    elif any([term[i].text.endswith(ne) for ne in nominals for i in range(len(term))]):
        term_type = "event"
    # POS = Verb implies event 
    elif any([t.pos_ == "VERB" for t in term]):
        term_type = "event"
    # default is otherwise entity 
    else:
        term_type = "entity"
    
    return term_type

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

def get_closest_match(indices1, indices2):
    """ Computes the closest pair of indices between two sets of indices.
    
    Each index pair corresponds to the start and end of a subsequence (i.e. phrases in a 
    sentence). Currently finds the closest pair just by checking the absolute distance between
    the start of each subsequence.
    
    Parameters
    ----------
    indices1: list of (int, int)
        List of index pairs corresponding to start and ends of subsequence 1
    indices2: list of (int, int)
        List of index pairs corresponding to start and ends of subsequence 2
    
    Returns
    -------
    tuple of ((int, int), (int, int))
        Pair of indices that are closest match 
        
    Examples
    --------

    >>> get_closest_match([(12, 15), (5, 6)], [(8, 10)])
    ((5, 6), (8, 10))
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

def write_spacy_docs(docs, vocab, filepath, vocab_filepath):
    """ Writes serialized spacy docs to file.  
    
    Parameters
    ----------
    docs: list of spacy.tokens.doc.Doc
        List of spacy Docs to write to file 
    filepath: str
        File path to serialized spacy docs
    """
    from spacy.attrs import IDS
    attr_exclude = ['SENT_START']
    attrs = [attr for attr in IDS.keys() if attr not in attr_exclude]
    
    doc_bin = DocBin(attrs=attrs)
    for doc in docs:
        doc_bin.add(doc)
        
    with open(filepath, 'wb') as f:
        f.write(doc_bin.to_bytes())
    with open(vocab_filepath, 'wb') as f:
        f.write(vocab.to_bytes())
    
def read_spacy_docs(filepath, vocab_filepath):
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
    from spacy.vocab import Vocab
    with open(vocab_filepath, 'rb') as f:
        vocab = Vocab().from_bytes(f.read())
        
    with open(filepath, 'rb') as f:
        data = f.read()
        
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(vocab))
    return docs
