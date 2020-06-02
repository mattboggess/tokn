from snorkel.labeling import labeling_function
from label_constants import *
import pickle

#===================================================================================

@labeling_function()
def has_pattern_lf(cand):
    """
    X [!not] has/have [!no] Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    while second.dep_ == 'conj':
        second = second.head
        if second == first:
            return ABSTAIN
    
    if second.head.text in ['has', 'have'] and second.head == first.head:
        if second.head.nbor(-1).text != 'not' and second.head.nbor(1).text != 'no':
            return label_classes.index('HAS-PART/REGION')
    
    return ABSTAIN

@labeling_function()
def consist_pattern_lf(cand):
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    while second.dep_ == 'conj':
        second = second.head
        if second == first:
            return ABSTAIN
    
    if second.head.text in ['consist', 'consisting', 'consists'] and \
       second.head == first.head:
        return label_classes.index('HAS-PART/REGION')
    
    return ABSTAIN

@labeling_function()
def contain_pattern_lf(cand):
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    while second.dep_ == 'conj':
        second = second.head
        if second == first:
            return ABSTAIN
    
    if second.head.text in ['contain', 'containing', 'contains'] and \
       second.head == first.head:
        return label_classes.index('HAS-PART/REGION')
    
    return ABSTAIN

@labeling_function()
def in_pattern_lf(cand):
    """
    X in the Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if second.head.text == 'in' and second.head.nbor(1).text in ['a', 'an', 'the'] and second.head.head == first:
        return label_classes.index('PART/REGION-OF')
    
    return ABSTAIN

@labeling_function()
def partof_pattern_lf(cand):
    """
    X in the Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_second = second.head.text == 'of' and second.head.head.text in ['parts', 'part']
    if valid_second:
        if second.head.head.head == first.head or second.head.head.head == first:
            return label_classes.index('PART/REGION-OF')
        
    return ABSTAIN

@labeling_function()
def of_pattern_lf(cand):
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if first.nbor(1).text == 'of' and first.nbor(2).text in ['a', 'an', 'the'] and \
       second.head == first.nbor(1) and not first.text.endswith('ion'):
        return label_classes.index('PART/REGION-OF')
    
    return ABSTAIN

@labeling_function()
def poss_pattern_lf(cand):
    second_start = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if first.nbor(1).text == "'s" and first.nbor(2) == second_start:
        return label_classes.index('HAS-PART/REGION')
    
    return ABSTAIN
    
    
@labeling_function()
def term_postmod_lf(cand):
    """
    Checks for modifier word added in front of shared base term: 
      i.e. daughter cell - hyponym - cell
    """
    term1_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]])
    term1_lemma = term1_lemma.replace(' - ', ' ')
    term2_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]])
    term2_lemma = term2_lemma.replace(' - ', ' ')
    term_pair = (term1_lemma, term2_lemma)
    
    term1 = term_pair[0].split(' ')
    term2 = term_pair[1].split(' ')
    
    # cell membrane PART/REGION-OF cell 
    if term1[0] == term2[0] and len(term2) < len(term1):
        return label_classes.index('PART/REGION-OF')
    elif term1[0] == term2[0] and len(term1) < len(term2):
        return label_classes.index('HAS-PART/REGION')
    else:
        return ABSTAIN

meronym_label_fns = [
    has_pattern_lf,
    in_pattern_lf,
    #term_postmod_lf,
    poss_pattern_lf,
    of_pattern_lf,
    contain_pattern_lf,
    consist_pattern_lf,
    partof_pattern_lf
]
