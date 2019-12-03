import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, defaultdict
import spacy

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

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
                       'type': ['Entity']}, 
         'biologist': {'text': ['biologist'], 'indices': [(1, 2)], 'pos': ['NN'], 
                       'type': ['Entity']}, 
         'cell': {'text': ['cell'], 'indices': [(7, 8)], 'tag': ['NN'], 'type': ['Entity']}}}
    """
    from spacy.lang.en.stop_words import STOP_WORDS
    spacy.tokens.token.Token.set_extension('workaround', default='', force=True)
    
    HEURISTIC_TOKENS = ["-", "plant", "substance", "atom"]
    
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
    found_terms = defaultdict(lambda: {"text": [], "indices": [], "pos": [], "type": []})
    
    # iterate through terms from longest to shortest
    terms = sorted(terms, key=len)[::-1]
    for spacy_term in terms:
        term_length = len(spacy_term)
        lemma_term_list = [token.lemma_ for token in spacy_term]
        text_term_list = [token.text for token in spacy_term]
        term_lemma = " ".join(lemma_term_list)
        
        # skip short acronyms that can cause problems
        if len(term_lemma) <= 2:
            continue
        
        # additional check to check for simple plural/single of uncommon biology terms
        match_uncommon_plural = lemma_term_list.copy()
        match_uncommon_plural[-1] = match_uncommon_plural[-1] + "s"

        # additional check using heuristics on lemmatized version
        match_heuristic = []
        if lemma_term_list[0] not in HEURISTIC_TOKENS:
            for token in lemma_term_list:
                if token not in HEURISTIC_TOKENS:
                    match_heuristic += token.split("-")
            heuristic_length = len(match_heuristic)
        else:
            heuristic_term = lemma_term_list
            heuristic_length = len(lemma_term_list)
        
        for ix in range(len(text) - term_length):
            
            heuristic_match = (lemmatized_text[ix:ix + heuristic_length] == match_heuristic)
            plural_match = (lemmatized_text[ix:ix + term_length] == match_uncommon_plural)
            lemma_match = (lemmatized_text[ix:ix + term_length] == lemma_term_list)
            text_match = (tokenized_text[ix:ix + term_length] == text_term_list)
            lower_match = ([t.lower() for t in tokenized_text[ix:ix + term_length]] ==
                           [t.lower() for t in text_term_list])
            
            # Only match on text if lemmatized version is a stop word (i.e. lower casing acronym)
            if term_lemma in STOP_WORDS:
                valid_match = text_match
            else:
                valid_match = heuristic_match or plural_match or text_match or lemma_match or lower_match
            
            if valid_match:
                
                if heuristic_match and not lemma_match:
                    match_length = heuristic_length
                else:
                    match_length = term_length
                
                term_text = " ".join([t.text for t in text[ix:ix + match_length]])
                term_tag = " ".join([t.tag_ for t in text[ix:ix + match_length]])
                
                # only tag term if not part of larger term
                if tags[ix:ix + match_length] == ["O"] * match_length:
                    
                    # classify term type
                    term_type = determine_term_type(spacy_term)
                    
                    # collect term information
                    found_terms[term_lemma]["text"].append(term_text)
                    found_terms[term_lemma]["indices"].append((ix, ix + match_length))
                    found_terms[term_lemma]["pos"].append(term_tag)
                    found_terms[term_lemma]["type"].append(term_type)
                    
                    # update sentence tags
                    tags = tag_bioes(tags, ix, match_length)
                    
                    # annotate token representations with term type
                    text[ix]._.workaround = f"<{term_type}>" + text[ix]._.workaround
                    end_ix = ix + match_length - 1
                    if text[end_ix]._.workaround.endswith(" "):
                        text[end_ix]._.workaround = text[end_ix]._.workaround[:-1] + f"</{term_type}> "
                    else:
                        text[end_ix]._.workaround += f"</{term_type}>"
                    
    # reconstruct fully annotated input text
    annotated_text = ""
    for token in text:
        annotated_text += token._.workaround
    
    return {
        "tokenized_text": tokenized_text, 
        "tags": tags, 
        "annotated_text": annotated_text,
        "found_terms": dict(found_terms)
    }

def merge_term_results(results1, results2, offset=0):
    for term in results2:
        if term not in results1:
            results1[term] = results2[term]
        else:
            for field in ["text", "indices", "pos", "type"]:
                if field == "indices":
                    indices_adj = [(t[0] + offset, t[1] + offset) for t in results2[term][field]]
                    results1[term][field] += indices_adj 
                else:
                    results1[term][field] += results2[term][field]
    return results1
    return results1

def postprocess_tagged_terms(tag_results, offset=0):
    """
    Takes in extracted terms and annotated text and updates the results by joining any consecutive
    singleton terms into a single term phrase.
    """
    
    term_lemmas = []
    term_texts = []
    term_indices = []
    term_types = []
    term_pos = []
    
    found_terms = tag_results["found_terms"]
    annotated_text = tag_results["annotated_text"]
    for term in found_terms:
        
        # flatten the dictionary
        term_lemmas += [term] * len(found_terms[term]["text"]) 
        term_texts += found_terms[term]["text"]
        term_indices += [(t[0] + offset, t[1] + offset) for t in found_terms[term]["indices"]]
        term_types += found_terms[term]["type"]
        term_pos += found_terms[term]["pos"]
    
    sort_ix = [i[0] for i in sorted(enumerate(term_indices), key=lambda x:x[1][0])]
    new_results = {}
    i = 0
    while i < len(sort_ix):
        ix = sort_ix[i]
        term_ix = term_indices[ix]
        start = term_ix[0]
        end = term_ix[1] 
        lemma = term_lemmas[ix] 
        text = term_texts[ix]
        pos = term_pos[ix]
        typ = term_types[ix]
        replace_text = f"<{typ}>{text}</{typ}>"
        
        while i + 1 < len(sort_ix):
            next_ix = sort_ix[i + 1]
            if term_indices[next_ix][0] == term_ix[1]:
                if "V" not in pos and "V" not in term_pos[next_ix]:
                    pos = f"{pos} {term_pos[next_ix]}"
                    text = f"{text} {term_texts[next_ix]}"
                    lemma = f"{lemma} {term_lemmas[next_ix]}"
                    typ = term_types[next_ix]
                    replace_text = f"{replace_text} <{typ}>{term_texts[next_ix]}</{typ}>"
                    term_ix = term_indices[next_ix]
                    ix = sort_ix[next_ix]
                    end = term_ix[1]
                    i += 1
                else:
                    break
                    
            # capture key term phrase with hyphen
            elif tag_results["tokenized_text"][term_ix[1]] == "-" and \
               term_indices[next_ix][0] == term_ix[1] + 1:
                if "V" not in pos and "V" not in term_pos[next_ix]:
                    pos = f"{pos} {term_pos[next_ix]}"
                    text = f"{text}-{term_texts[next_ix]}"
                    lemma = f"{lemma}-{term_lemmas[next_ix]}"
                    typ = term_types[next_ix]
                    replace_text = f"{replace_text}-<{typ}>{term_texts[next_ix]}</{typ}>"
                    term_ix = term_indices[next_ix]
                    ix = sort_ix[next_ix]
                    end = term_ix[1]
                    i += 1
                else:
                    break
            else:
                break
                    
        # merging terms
        if lemma in new_results:
            new_results[lemma]["type"].append(typ)
            new_results[lemma]["text"].append(text)
            new_results[lemma]["pos"].append(pos)
            new_results[lemma]["indices"].append((start, end))
        else:
            new_results[lemma] = {
                "type": [typ],
                "text": [text],
                "indices": [(start, end)],
                "pos": [pos]}

        annotated_text = annotated_text.replace(replace_text, 
                                                f"<{typ}>{text}</{typ}>")
                    
        i += 1
    
    return {"found_terms": new_results, "annotated_text": annotated_text}
        
def determine_term_type(term):
    """ Categorizes a term as either entity or event based on several derived rules.

    Parameters
    ----------
    term: spacy.tokens.doc.Doc
        Spacy preprocessed representation of the term 

    Returns
    -------
    str ('entity' | 'event')
        The class of the term
    """
    
    NOMINALS = ["ation", "ition", "ption", "ing", "sis", "lism", "ment", "sion"]
    EVENT_KEYWORDS = ["process", "cycle"]
    
    # key words that indicate events despite being nouns 
    if any([ek in term.text.lower() for ek in EVENT_KEYWORDS]):
        term_type = "event"
    # key endings indicating a nominalized form of an event 
    elif any([term[i].text.endswith(ne) for ne in NOMINALS for i in range(len(term))]):
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
