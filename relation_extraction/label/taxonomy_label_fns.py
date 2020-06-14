from snorkel.labeling import labeling_function
from label_constants import *
import pickle

#===================================================================================
# Sentence Dependency Pattern-Based Taxonomy Labelers

# Hearst 1992 Patterns

@labeling_function()
def suchas_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X such as Y1, Y2, ..., and/or Yk 
      - Y1, ..., Yk -> SUBCLASS -> X
      - X -> SUPERCLASS -> Y1, ..., Yk
      
    One of original Hearst patterns (Hearst, 1992)
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    # follow the chain of conjunctions back to the as
    while start.text != 'as':
        if start.dep_ == 'conj' or start.dep_ == 'pobj':
            start = start.head
        else:
            break
    
    if start.text == 'as':
        if 'such' in [ch.text for ch in start.children]:
            if start.head.idx == end.idx:
                if cand.term1_location[0] < cand.term2_location[0]:
                    return label_classes.index('SUPERCLASS')
                else:
                    return label_classes.index('SUBCLASS')
    return ABSTAIN

@labeling_function()
def including_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X including Y1, Y2, ..., and/or Yk 
      - Y -> SUBCLASS -> X1, X2, ...
      - X -> SUPERCLASS -> Y1, ..., Yk
      
    One of original Hearst patterns (Hearst, 1992)
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    # follow the chain of conjunctions back to the as
    while second.text != 'including':
        if second.dep_ == 'conj' or second.dep_ == 'pobj':
            second = second.head
        else:
            break
            
    second_valid = second.text == 'including'
    if second_valid and (cand.term1_location[1] - 1) < (len(cand.doc) - 3):
        first_valid = first.nbor(2) == second 
    else:
        first_valid = False
    
    if first_valid and second_valid:
        if cand.term1_location[0] < cand.term2_location[0]:
            return label_classes.index('SUPERCLASS')
        else:
            return label_classes.index('SUBCLASS')
    return ABSTAIN

@labeling_function()
def especially_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: Y especially X1, X2, ..., and/or Xk 
      - X -> SUPERCLASS -> Y1, ..., Yk
      - Y -> SUBCLASS -> X1, ..., Xk
      
    One of original Hearst patterns (Hearst, 1992)
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    # follow the chain of conjunctions back to the esepcially
    while second.idx > 1 and second.nbor(-1).text != 'especially':
        if second.dep_ == 'conj' or second.dep_ == 'appos':
            second = second.head
        else:
            break
        
    second_valid = second.idx > 1 and second.nbor(-1).text == 'especially'
    if second_valid:
        first_valid = first.nbor(2) == second.nbor(-1)
    else:
        first_valid = False 
    
    if first_valid and second_valid: 
        if cand.term1_location[0] < cand.term2_location[0]:
            return label_classes.index('SUPERCLASS')
        else:
            return label_classes.index('SUBCLASS')

    return ABSTAIN

@labeling_function()
def other_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X1, X2, ..., and/or other Y 
      - Label: X1, X2, ... -> SUBCLASS -> Y
      
    One of original Hearst patterns (Hearst, 1992)
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    start_first = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    
    while start != end:
        if start.dep_ == 'conj':
            start = start.head
        else:
            break
            
    if start.text == end.text and \
       cand.term2_location[0] > 2 and \
       start_first.nbor(-2).text in ['and', 'or'] and \
       start_first.nbor(-1).text == 'other':
        if cand.term1_location[0] < cand.term2_location[0]:
            return label_classes.index('SUBCLASS')
        else:
            return label_classes.index('SUPERCLASS')
    
    return ABSTAIN

# Snow et al. 2004 Patterns

@labeling_function()
def called_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: Y called X
      - Y -> SUPERCLASS -> X
      
    One of highlighted patterns from Snow et al. 2004
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if start.head.text == 'called' and start.dep_ == 'oprd':
        if start.head.dep_ == 'acl' and start.head.head.text == end.text:
            if cand.term1_location[0] < cand.term2_location[0]:
                return label_classes.index('SUPERCLASS')
            else:
                return label_classes.index('SUBCLASS')
    return ABSTAIN

@labeling_function()
def isa_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X is a/an/the Y
      - X -> SUBCLASS -> Y
      - Y -> SUPERCLASS -> X
      
    One of highlighted patterns from Snow et al. 2004
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    # second term is directly linked to an is 
    valid_second = second.dep_ == 'attr' and second.head.text == 'is'
    
    # first term is linked to an are either directly or via conjunction 
    while first.text != 'is':
        if first.dep_ == 'conj' or first.dep_ == 'nsubj':
            first = first.head
        else:
            break
    valid_first = first == second.head 
    
    if valid_second and valid_first:
        if cand.term1_location[0] < cand.term2_location[0]:
            return label_classes.index('SUBCLASS')
        else:
            return label_classes.index('SUPERCLASS')
        
    return ABSTAIN

@labeling_function()
def appo_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X, a/an Y (appos)
      - X -> SUBCLASS -> Y
      
    One of highlighted patterns from Snow et al. 2004
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    second_conj = len([ch for ch in second.children if ch.dep_ == 'conj']) > 0
    valid_second = second.head == first and second.dep_ == 'appos' and not second_conj
    
    if valid_second:
        valid_first = first.nbor(1).text in [',', '('] and \
                      first.nbor(2).text in ['a', 'an', 'the', 'e.g.', 'i.e.'] and \
                      first.nbor(2) in [ch for ch in second.children]
    else:
        valid_first = False
    
    if valid_second and valid_first:
        if cand.term1_location[0] < cand.term2_location[0]:
            if first != cand.doc[0] and first.nbor(-1).text in ['some', 'another', 'specialized']:
                return label_classes.index('SUPERCLASS')
            else:
                return label_classes.index('SUBCLASS')
        else:
            return label_classes.index('SUPERCLASS')
    return ABSTAIN

# Custom Patterns

@labeling_function()
def are_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern1: X1, ..., Xk are Y (X -> SUBCLASS -> Y)
      - Pattern2: The X are Y (X -> SUPERCLASS -> Y)
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first_children = [tok.text.lower() for tok in first.children]
    
    # second term is directly linked to an are
    valid_second = second.dep_ == 'attr' and second.head.text == 'are'
    
    # first term is linked to an are either directly or via conjunction 
    while first.text != 'are':
        if first.dep_ == 'conj' or first.dep_ == 'nsubj':
            first = first.head
        else:
            break
    valid_first = first == second.head
    
    if valid_second and valid_first and second.head.nbor(-1).text != 'not':
        if 'the' in first_children:
            if cand.term1_location[0] < cand.term2_location[0]:
                return label_classes.index('SUPERCLASS')
            else:
                return label_classes.index('SUBCLASS')
        else:
            if cand.term1_location[0] < cand.term2_location[0]:
                return label_classes.index('SUBCLASS')
            else:
                return label_classes.index('SUPERCLASS')
        
    return ABSTAIN

@labeling_function()
def symbolconj_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X (Y1, Y2, ... and/or Yk)
      - X -> SUPERCLASS -> Y
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    second_conj = len([ch for ch in second.children if ch.dep_ == 'conj']) > 0
    second_start = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    
    valid_first = first.nbor(1).text in ['(', '-', '—', ':']
    if valid_first:
        if not second_conj and second_start.nbor(-1) == first.nbor(1):
            return ABSTAIN
        while second.dep_ == 'conj':
            if second.dep_ == 'conj':
                second = second.head
            else:
                break
        if (second != cand.doc[0] and second.nbor(-1) == first.nbor(1)):
            return label_classes.index('SUPERCLASS')
        
    return ABSTAIN

#===================================================================================
# Sentence Dependency Pattern-Based Synonym Labelers

@labeling_function()
def parens_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X, (Y) (X -> SYNONYM -> Y)
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    second_start = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if second == cand.doc[-1]:
        return ABSTAIN
    
    valid_first = first.nbor(1).text == '('
    valid_second = second_start.nbor(-1) == first.nbor(1) and \
                   second.nbor(1).text == ')'
                    
    if valid_second and valid_first:
        return label_classes.index('SYNONYM')
    
    return ABSTAIN

@labeling_function()
def also_knownas_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X also known as Y (X -> SYNONYM -> Y)
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_second = cand.term2_location[1] > 3 and \
                   second.head.text == 'as' and \
                   second.head.nbor(-1).text == 'known' and \
                   second.head.nbor(-2).text == 'also'
    
    valid_first = False
    if valid_second:
        valid_first = (first.head == second.head.nbor(-1)) or (second.head.nbor(-1) in first.children)
    
    if valid_first and valid_second:
        return label_classes.index('SYNONYM')
    
    return ABSTAIN

@labeling_function()
def also_called_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X also called Y (X -> SYNONYM -> Y)
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_second = cand.term2_location[1] > 3 and \
                   second.head.text == 'called' and \
                   second.head.nbor(-1).text == 'also'
    
    valid_first = False
    if valid_second:
        valid_first = (second.head.nbor(-1) in first.children) or (first.head == second.head)
    
    if valid_first and valid_second:
        return label_classes.index('SYNONYM')
    
    return ABSTAIN

@labeling_function()
def plural_pattern_lf(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X (plural/singular [=] Y) (X -> SYNONYM -> Y)
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    second_start = cand.doc[max(cand.term1_location[0], cand.term2_location[0])]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_first = first.nbor(1).text == '(' and first.nbor(2).text in ['plural', 'singular']
    valid_second = second.head == first and \
                   second.dep_ == 'appos' and \
                   second_start.nbor(-1).text in ['=', 'plural', 'singular'] and \
                   second.nbor(1).text == ')'
                    
    if valid_second and valid_first:
        return label_classes.index('SYNONYM')
    
    return ABSTAIN

#===================================================================================
# Term-Based Heuristic Labelers

@labeling_function()
def term_modifier_lf(cand):
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
    
    # daughter cell SUBCLASS cell 
    if term1[-1] == term2[-1] and len(term2) < len(term1):
        return label_classes.index('SUBCLASS')
    # cell SUPERCLASS daughter cell 
    elif term1[-1] == term2[-1] and len(term1) < len(term2):
        return label_classes.index('SUPERCLASS')
    else:
        return ABSTAIN

@labeling_function()
def term_subset_lf(cand):
    """
    Checks for term is modified version of base term with same root: oncogene - hyponym - gene 
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
    
    # gene SUPERCLASS oncogene
    if term2[-1].endswith(term1[-1]) and len(term1) == 1 and len(term2) == 1:
        return label_classes.index('SUPERCLASS')
    # oncogene SUBCLASS gene
    elif term1[-1].endswith(term2[-1]) and len(term2) == 1 and len(term1) == 1:
        return label_classes.index('SUBCLASS')
    else:
        return ABSTAIN
    
taxonomy_label_fns = [
    isa_pattern_lf,
    suchas_pattern_lf,
    including_pattern_lf,
    called_pattern_lf,
    especially_pattern_lf,
    appo_pattern_lf, 
    other_pattern_lf, 
    are_pattern_lf,
    symbolconj_pattern_lf,
    term_modifier_lf,
    term_subset_lf,
    also_knownas_pattern_lf,
    parens_pattern_lf,
    also_called_pattern_lf,
    plural_pattern_lf
]
