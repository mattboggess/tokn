import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from collections import defaultdict
import itertools
import numpy as np


def get_closest_match(indices1, indices2):
    """
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
     
      

# def tag_text(text, terms, relations=None, nlp=None):
#     """ Identifies and tags any terms and pairwise relations between terms present in a given 
#     input text.

#     Searches through the input text and finds all terms (single words and phrases) that are present
#     in the list of provided terms. Then searches through each pair of found terms to determine
#     if any relations between those terms exist in the provided relations structure. 
#     Returns a list of found terms & relations with indices and POS tagging as well as a 
#     BIOES tagged version of the sentence denoting where the terms are in the sentences. 
    
#     Uses spacy functionality to tokenize and lemmatize for matching text (it is recommended to 
#     preprocess by Spacy before inputting to prevent repeated work if calling multiple times).

#     Gives precedence to longer terms first so that terms that are part of a larger term phrase
#     are ignored (i.e. match 'cell wall', not 'cell' within the phrase cell wall). Pairs of terms
#     that do not have an existing relation between them get tagged with the special 'no-relation'
#     relation.

#     Parameters
#     ----------
#     text: str | spacy.tokens.doc.Doc
#         Input text that will be/are preprocessed using spacy and searched for terms
#     terms: list of str | list of spacy.tokens.doc.Doc
#         List of input terms that will be/are preprocessed using spacy. 
#     nlp: 
#         Spacy nlp pipeline object that will tokenize, POS tag, lemmatize, etc. 

#     Returns
#     -------
#     dict 
#         'found_terms': list of found terms each with list of indices where matches were found and
#         basic part of speech information
#         Second element: tokenized text as list of tokens
#         Third element list of BIOES tags for the tokenized text

#     Examples
#     --------

#     >>> tag_text('A biologist will tell you that a cell contains a cell wall.', 
#                  ['cell', 'cell wall', 'biologist'], [('cell', 'has-part', 'cell wall')])
#     ([{'cell wall': [1]}], 
#       ['The', 'cell', 'wall', 'provides', 'structure', '.'], 
#       ['O', 'B', 'E', 'O', 'O', 'O'])

#     """

#     # default to Stanford NLP pipeline wrapped in Spacy
#     if nlp is None:
#         snlp = stanfordnlp.Pipeline(lang="en")
#         nlp = StanfordNLPLanguage(snlp)
        
#     # preprocess with spacy if needed
#     if type(terms[0]) != spacy.tokens.doc.Doc:
#         terms = [nlp(term) for term in terms]
#     if type(text) != spacy.tokens.doc.Doc:
#         text = nlp(text)

#     normalized_text = [token.lemma_ for token in text]
#     tokenized_text = [token.text for token in text]

#     # storage variables
#     tagged_text = ['O'] * len(text)
#     found_terms = defaultdict(lambda: {"text": [], "indices": [], "tag": []})

#     # iterate through terms from longest to shortest
#     terms = sorted(terms, key=len)[::-1]
#     for term in terms:
#         normalized_term = [token.lemma_ for token in term]

#         for ix in range(len(text) - len(term)):

#             if normalized_text[ix:ix + len(term)] == normalized_term:
#                 # only add term if not part of larger term
#                 if tagged_text[ix:ix + len(term)] == ["O"] * len(term):
#                     lemma = " ".join(normalized_term)
#                     found_terms[lemma]["text"].append(" ".join([token.text for token in term]))
#                     found_terms[lemma]["indices"].append(ix)
#                     found_terms[lemma]["tag"].append(" ".join([token.tag_ for token in term]))
#                     tagged_text = tag_bioes(tagged_text, ix, len(term))
    
#     # extract relations
#     if relations:
#         found_relations = tag_relations(sorted(list(found_terms.keys())), relations)
#     else:
#         found_relations = []
    
#     output = {
#         "input_text": str(text),
#         "tokenized_text": tokenized_text,
#         "tagged_text": tagged_text,
#         "relations": found_relations,
#         "terms": dict(found_terms)
#     }
#     return output 


# def tag_relations(terms, relations):
#     """ Produces all pairwise relations between a set of terms given a set of relations.
    
#     Each term pair will either be labeled with one or more relation existing in the set of 
#     input relations or will be labeled with the 'no-relation' label.
    
#     Parameters
#     ----------
#     terms: list of str
#         List of terms for which we want to assign pairwise relations.
#     relations: list of tuple
#         List of relation triplets consisting of (term1, relation-type, term2)
    
#     Returns
#     -------
#     list of tuple
#         List of relation triplets present within the passed terms
    
#     Examples
#     --------
    
    
    
#     """
#     found_relations = []
#     for i in range(len(terms) - 1):
#         for j in range(i + 1, len(terms)):
#             # sort alphabetically so no-relations will have same ordering across all texts 
#             term1, term2 = sorted([terms[i], terms[j]])
#             relation_count = 0
#             for relation in relations:
#                 if set([term1, term2]) == set([relation[0], relation[2]]):
#                     found_relations.append(relation)
#                     relation_count += 1
#             if relation_count == 0:
#                 found_relations.append((term1, "no-relation", term2))
    
#     return found_relations


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
