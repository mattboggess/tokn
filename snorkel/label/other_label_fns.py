from snorkel.labeling import labeling_function
from label_constants import *

@labeling_function()
def nsubj_pattern(cand):
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    if first.dep_ == 'nsubj' and second.dep_ == 'nsubj':
        return OTHER
    return ABSTAIN

@labeling_function()
def term_part_of_speech(cand):
    """
    Entities must be nouns/noun phrases. Check the part of speech of each of the
    terms to filter out verbs and other terms that aren't valid.
    """
    invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 
                   'VBN', 'VBZ', 'VBP', 'WRB']
    
    term1_pos = [tok.tag_ for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]]
    term2_pos = [tok.tag_ for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]]
    
    if term1_pos[-1] in invalid_pos or term2_pos[-1] in invalid_pos:
        return OTHER
    else:
        return ABSTAIN
    
@labeling_function()
def term_dep_role(cand):
    """
    Entities must be the root of noun phrases, not compound nouns or other noun
    modifiers (i.e. don't match 'cell' if 'cell membrane' not in term list)
    """
    invalid_dep = ['npadvmod', 'compound', 'poss', 'amod']
    
    term1_dep = [tok.dep_ for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]]
    term2_dep = [tok.dep_ for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]]
    
    if term1_dep[-1] in invalid_dep or term2_dep[-1] in invalid_dep:
        return OTHER
    else:
        return ABSTAIN