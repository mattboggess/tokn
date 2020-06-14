from snorkel.labeling import labeling_function
from label_constants import *
import pickle

#===================================================================================

@labeling_function()
def has_pattern_lf(cand):
    """
    X [!not] has/have [!no] Y -> X HAS-PART/REGION Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if first.text in ['structure', 'function', 'structures', 'functions'] or \
       second.text in ['structure', 'function', 'structures', 'functions']:
        return ABSTAIN
    
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
    """
    X consists/consisting of Y -> X HAS-PART/REGION Y
    """
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
    """
    X contains/containing Y -> X HAS-PART/REGION Y
    """
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
    X in the Y -> X PART/REGION OF Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if second.head.text == 'in' and second.head.nbor(1).text in ['a', 'an', 'the'] and second.head.head == first:
        return label_classes.index('PART/REGION-OF')
    
    return ABSTAIN

@labeling_function()
def partof_pattern_lf(cand):
    """
    X [is/are] part of Y -> X PART/REGION-OF Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_second = second.head.text == 'of' and second.head.head.text in ['parts', 'part']
    if valid_second:
        if second.head.head.head == first.head or second.head.head.head == first:
            return label_classes.index('PART/REGION-OF')
        
    return ABSTAIN

@labeling_function()
def poss_pattern_lf(cand):
    """
    X's Y -> X HAS-PART/REGION Y
    """
    second_start = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if first.nbor(1).text == "'s" and first.nbor(2) == second_start:
        return label_classes.index('HAS-PART/REGION')
    
    return ABSTAIN
    
    
meronym_label_fns = [
    has_pattern_lf,
    in_pattern_lf,
    poss_pattern_lf,
    contain_pattern_lf,
    consist_pattern_lf,
    partof_pattern_lf
]
