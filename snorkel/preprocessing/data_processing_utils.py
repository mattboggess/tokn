from spacy.tokens import DocBin
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import spacy
import numpy as np
import itertools
from collections import defaultdict

def match_heuristic(text_span, term_span):
    """
    Given a span of lemmatized text and a lemmatized term, match the two spans with several 
    heuristics to account for slight mismatches. This includes mapping special characters.
    """
    HEURISTIC_MAPPING = {
        'alpha': 'α',
        'beta': 'β',
        'prime': '′'
    }
    
    # remap alternate representations
    text_span = [HEURISTIC_MAPPING[tok] if tok in HEURISTIC_MAPPING else tok for tok in text_span]
    term_span = [HEURISTIC_MAPPING[tok] if tok in HEURISTIC_MAPPING else tok for tok in term_span]
    
    return term_span == text_span

def match_uncommon_plurals(text_span, term_span):
    """
    Tries some simple modifications on the lemmatized term to match plurals that aren't handled
    correctly.
    """
    s_plural = term_span.copy()
    s_plural[-1] = s_plural[-1] + 's'
    s_plural_match = s_plural == text_span
    
    es_plural = term_span.copy()
    es_plural[-1] = es_plural[-1] + 'es'
    es_plural_match = es_plural == text_span
    
    return s_plural_match or es_plural_match

def tag_terms(text, terms, nlp=None, invalid_pos=[], invalid_dep=[]):
    """ Identifies and tags any terms in a given input text.
    
    TODO:
      - Handle pronoun lemmas more properly
      - Handle noun chunk partials more fully (expand if subset??)

    Searches through the input text and finds all terms (single words and phrases) that are present
    in the list of provided terms. Returns a list of found terms with indices and POS/Dependency 
    parse tags from Spacy as well as a BIOES tagged version of the sentence denoting where the terms 
    are in the sentences. 
  
    Uses spacy functionality to tokenize and lemmatize for matching text (it is recommended to 
    preprocess by Spacy before inputting to prevent repeated work if calling multiple times).

    Gives precedence to longer terms first so that terms that are part of a larger term phrase
    are ignored (i.e. match 'eukaryotic cell', not 'cell' within the phrase eukaryotic cell). 

    Parameters
    ----------
    text: str | spacy.tokens.doc.Doc
        Input text that will be/are preprocessed using spacy and searched for terms
    terms: list of str | list of spacy.tokens.doc.Doc
        List of input terms to tag that will be/are preprocessed using spacy. 
    nlp: 
        Spacy nlp pipeline object that will tokenize, POS tag, lemmatize, etc. 
    invalid_pos: list of str
        List of Spacy part of speech tags that should be ignored when tagging. 
    invalid_dep: list of str
        List of Spacy dependency parse tags that should be ignored when tagging. Useful for 
        restricting tagged terms to the root of a noun phrase for example.

    Returns
    -------
    dict with four entries: 
        tokenized_text: tokenized text as list of tokens
        tags: list of BIOES tags for the tokenized text
        annotated_text: original text with <term> tags put around found terms in the input text
        found_terms: list of found terms as dictionaries mapping lemmatized form of term to 
        list of indices where matches were found and other information such as part of speech, 
        dependency parse role, etc.

    Examples
    --------
    >>> tag_text('A biologist will tell you that a cell contains a cell wall.', 
                 ['cell', 'cell wall', 'biologist'])
                 
    {
        'annotated_text': "A <term>biologist</term> will tell you that a <term>cell</term> contains a <term>cell wall</term>.,
        'tokenized_text': ['A', 'biologist', 'will', 'tell', 'you', 'that', 'a', 'cell', 'contains', 'a', 'cell', 'wall', '.'],
        'found_terms': {
            'biologist': {
                'text': ['biologist'], 
                'tokens': [['biologist']], 
                'pos': [['NN']], 
                'dep': [['nsubj']], 
                'indices': [(1, 2)]
            },
            'cell': {
                'text': ['cell'], 
                'tokens': [['cell']],
                'pos': [['NN']], 
                'dep': [['nsubj']], 
                'indices': [(7, 8)]
            },
            'cell wall': {
                'text': ['cell wall'], 
                'tokens': [['cell', 'wall']],
                'pos': [['NN', 'NN']], 
                'dep': [['compound', 'dobj']], 
                'indices': [(10, 12)]}
        },
        'bioes_tags': ['O', 'S', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'B', 'E', 'O']
    }
   """
    from spacy.lang.en.stop_words import STOP_WORDS
    spacy.tokens.token.Token.set_extension('workaround', default='', force=True)
    
    # preprocess with spacy if needed
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    if type(terms[0]) != spacy.tokens.doc.Doc:
        terms = [nlp(term) for term in terms]
    if type(text) != spacy.tokens.doc.Doc:
        text = nlp(text)
        
    # set up a custom representation of the text in spacy where we can add term type annotations
    for token in text:
        token._.workaround = token.text_with_ws
    
    lemmatized_text = [token.lemma_.lower() for token in text]
    tokenized_text = [token.text for token in text]
    tags = ['O'] * len(text)
    found_terms = defaultdict(lambda: {'text': [], 'tokens': [], 'indices': [], 'pos': [], 'dep': []})
    
    # iterate through terms from longest to shortest to ensure we tag the largest possible phrase
    terms = sorted(terms, key=len)[::-1]
    for spacy_term in terms:
        
        # lemma representation of term (ignoring hyphens and case)
        lemma_term_span = [token.lemma_.lower() for token in spacy_term if token.lemma_ != '-']
        term_length = len(lemma_term_span)
        term_lemma = ' '.join(lemma_term_span)
        
        # exact text representation of term
        text_term_span = [token.text for token in spacy_term]
        term_text = ''.join([token.text_with_ws for token in spacy_term])
        
        # skip terms with stop word equivalent representation
        if term_text in STOP_WORDS or term_lemma in STOP_WORDS:
            continue
            
        # skip words that get lemmatized to pronoun and weren't caught by stop words check
        # TODO: Handle this more properly
        if term_lemma == '-pron-':
            continue
        
        # check all subsequences of the same length as the term for a match
        for ix in range(len(text) - term_length):
            
            text_span = lemmatized_text[ix:ix + term_length]
            
            # handle hyphens by extending terms beyond them so we can ignore them
            num_hyphens = len([tok for tok in text_span if tok == '-'])
            text_span += lemmatized_text[ix + term_length:min(len(text), 
                                                              ix + term_length + num_hyphens)] 
            match_length = len(text_span)
            text_span = [tok for tok in text_span if tok != '-']
            
            # match directly on lemma
            lemma_match = text_span == lemma_term_span
            # match directly on text 
            text_match = text_span == text_term_span
            # match with a few heuristic rules/term substitutions (sort of a catch all)
            heuristic_match = match_heuristic(text_span, lemma_term_span)
            # try to match some uncommon plurals that Spacy doesn't lemmatize correctly
            plural_match = match_uncommon_plurals(text_span, lemma_term_span) 
            # good to go if any match
            valid_match = heuristic_match or plural_match or text_match or lemma_match 
            
            # only tag term if not part of larger term
            if valid_match and tags[ix:ix + match_length] == ['O'] * match_length:
                    
                # collect term information
                term_text = ''.join([token.text_with_ws for token in text[ix:ix + match_length]]).strip()
                term_tokens = [t.text for t in text[ix:ix + match_length]]
                term_tag = [t.tag_ for t in text[ix:ix + match_length]]
                term_dep = [t.dep_ for t in text[ix:ix + match_length]]
                
                # screen for invalid dependencies and pos tags
                if term_tag[-1] in invalid_pos or term_dep[-1] in invalid_dep:
                    continue
                
                found_terms[term_lemma]['text'].append(term_text)
                found_terms[term_lemma]['tokens'].append(term_tokens)
                found_terms[term_lemma]['indices'].append((ix, ix + match_length))
                found_terms[term_lemma]['pos'].append(term_tag)
                found_terms[term_lemma]['dep'].append(term_dep)
                    
                # update sentence tags
                tags = tag_bioes(tags, ix, match_length)

                # annotate input text with term markings
                text[ix]._.workaround = '<term>' + text[ix]._.workaround
                end_ix = ix + match_length - 1
                if text[end_ix]._.workaround.endswith(' '):
                    text[end_ix]._.workaround = text[end_ix]._.workaround[:-1] + '</term>'
                else:
                    text[end_ix]._.workaround += '</term>'
                    
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
