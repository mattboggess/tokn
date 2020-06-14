from snorkel.labeling import labeling_function
from label_constants import *

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
        return label_classes.index('OTHER')
    else:
        return ABSTAIN
    