from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.labeling import labeling_function
import pickle

# Relation Classes
HYPONYM = 1 # subclass-of
HYPERNYM = 2 # superclass-of
OTHER = 0
ABSTAIN = -1

#===================================================================================
# Sentence Dependency Pattern-Based Positive Labelers

# Hearst 1992 Patterns

@labeling_function()
def suchas_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X such as Y1, Y2, ..., and/or Yk 
      - Y1, ..., Yk -> HYPONYM -> X
      - X -> HYPERNYM -> Y1, ..., Yk
      
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
                    return HYPERNYM
                else:
                    return HYPONYM 
    return ABSTAIN

@labeling_function()
def including_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X including Y1, Y2, ..., and/or Yk 
      - Y -> HYPONYM -> X1, X2, ...
      - X -> HYPERNYM -> Y1, ..., Yk
      
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
    if second_valid:
        first_valid = first.nbor(2) == second 
    else:
        first_valid = False
    
    if first_valid and second_valid:
        if cand.term1_location[0] < cand.term2_location[0]:
            return HYPERNYM
        else:
            return HYPONYM 
    return ABSTAIN

@labeling_function()
def especially_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: Y especially X1, X2, ..., and/or Xk 
      - X -> HYPERNYM -> Y1, ..., Yk
      - Y -> HYPONYM -> X1, ..., Xk
      
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
            return HYPERNYM 
        else:
            return HYPONYM

    return ABSTAIN

@labeling_function()
def other_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X1, X2, ..., and/or other Y 
      - Label: X1, X2, ... -> HYPONYM -> Y
      
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
       start_first.nbor(-2).text in ['and', 'or'] and \
       start_first.nbor(-1).text == 'other':
        if cand.term1_location[0] < cand.term2_location[0]:
            return HYPONYM 
        else:
            return HYPERNYM 
    
    return ABSTAIN

# Snow et al. 2004 Patterns

@labeling_function()
def called_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: Y called X
      - Y -> HYPERNYM -> X
      
    One of highlighted patterns from Snow et al. 2004
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    if start.head.text == 'called' and start.dep_ == 'oprd':
        if start.head.dep_ == 'acl' and start.head.head.text == end.text:
            if cand.term1_location[0] < cand.term2_location[0]:
                return HYPERNYM 
            else:
                return HYPONYM 
        elif start.head.nbor(-1).text in ['also', 'sometimes']: # also/sometimes called implies synonym
            return OTHER
    return ABSTAIN

@labeling_function()
def isa_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X is a/an/the Y
      - X -> HYPONYM -> Y
      - Y -> HYPERNYM -> X
      
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
            return HYPONYM
        else:
            return HYPERNYM
        
    return ABSTAIN

@labeling_function()
def appo_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X, a/an Y (appos)
      - X -> HYPONYM -> Y
      
    One of highlighted patterns from Snow et al. 2004
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    second_conj = len([ch for ch in second.children if ch.dep_ == 'conj']) > 0
    valid_second = second.head == first and second.dep_ == 'appos' and not second_conj
    
    if valid_second:
        valid_first = first.nbor(1).text == ',' and \
                      first.nbor(2).text in ['a', 'an'] and \
                      first.nbor(2) in [ch for ch in second.children]
    else:
        valid_first = False
    
    if valid_second and valid_first:
        if cand.term1_location[0] < cand.term2_location[0]:
            return HYPONYM
        else:
            return HYPERNYM 
    return ABSTAIN

# Custom Patterns

@labeling_function()
def are_pattern(cand):
    """
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern1: X1, ..., Xk are Y (X -> HYPONYM -> Y)
      - Pattern2: The X are Y (X -> HYPERNYM -> Y)
      
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
                return HYPERNYM
            else:
                return HYPONYM
        else:
            if cand.term1_location[0] < cand.term2_location[0]:
                return HYPONYM
            else:
                return HYPERNYM
        
    return ABSTAIN

@labeling_function()
def whichis_pattern(cand):
    """
    TODO: FIX/ADD (Not many matches)
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X which is a/an/the Y 
      - X -> HYPONYM -> Y
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    valid_second = second.head.text == 'is' and second.dep_ == 'attr' 
    if valid_second:
        valid_first = second.head.head.text == first.text and second.head.nbor(-1).text == 'which'
    else:
        valid_first = False
    
    if valid_second and valid_first:
        if cand.term1_location[0] < cand.term2_location[0]:
            return HYPONYM
        else:
            return HYPERNYM
    
    return ABSTAIN

@labeling_function()
def knownas_pattern(cand):
    """
    TODO: FIX/ADD (Not many matches)
    Matches sentence structure pattern using Spacy dependency parse:
      - Pattern: X [!also] known as Y 
      - X -> HYPONYM -> Y
      - Y -> HYPERNYM -> X
    If also precedes the known as this is classified as OTHER since this implies an
    alternate name for the same entity.
      
    Custom added pattern from scanning textbook sentences for patterns 
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    also_flag = False
    known_flag = False
    while start != end:
        if start.dep_ == 'pobj' and start.head.text == 'as':
            start = start.head
        elif start.dep_ == 'prep' and start.head.text == 'known':
            start = start.head
        elif start.text == 'known':
            also_flag == start.nbor(-1).text == 'also'
            known_flag = True
            start = start.head
            break
        else:
            break
            
    if start.text == end.text and known_flag:
        if also_flag:
            return OTHER
        elif cand.term1_location[0] < cand.term2_location[0]:
            return HYPERNYM 
        else:
            return HYPONYM 
    
    return ABSTAIN

#===================================================================================
# Sentence Dependency Pattern-Based Negative Labelers

@labeling_function()
def list_pattern(cand):
    """
    FIX THIS!
    """
    start = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    end = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    
    # follow the chain of conjunctions back to the as
    while start != end:
        if start.dep_ == 'conj':
            start = start.head
        else:
            break
    
    if start.text == end.text:
        return OTHER

    return ABSTAIN

@labeling_function()
def nsubj_pattern(cand):
    second = cand.doc[max(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    first = cand.doc[min(cand.term1_location[1] - 1, cand.term2_location[1] - 1)]
    if first.dep_ == 'nsubj' and second.dep_ == 'nsubj':
        return OTHER
    return ABSTAIN
    
#===================================================================================
# Term-Based Labelers

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
    
@labeling_function()
def term_modifier(cand):
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
    
    # daughter cell HYPONYM cell 
    if term1[-1] == term2[-1] and len(term2) < len(term1):
        return HYPONYM
    # cell HYPERNYM daughter cell 
    elif term1[-1] == term2[-1] and len(term1) < len(term2):
        return HYPERNYM 
    else:
        return ABSTAIN

@labeling_function()
def term_subset(cand):
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
    
    # gene HYPERNYM oncogene
    if term2[-1].endswith(term1[-1]) and len(term1) == 1 and len(term2) == 1:
        return HYPERNYM
    # oncogene HYPONYM gene
    elif term1[-1].endswith(term2[-1]) and len(term2) == 1 and len(term1) == 1:
        return HYPONYM
    else:
        return ABSTAIN

#===================================================================================
# Distant Supervision Labelers

with open("../data/kb_bio101_relations_db.pkl", 'rb') as fid:
    relations = pickle.load(fid)
with open("../data/kb_bio101_terms.pkl", 'rb') as fid:
    terms = pickle.load(fid)

@labeling_function(resources=dict(relations=relations, terms=terms))
def kb_bio101_ds_taxonomy(cand, terms, relations):
    """
    Looks up term pair KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a HYPONYM/HYPERNYM label depending
    on term pair ordering. If it finds both terms in the KB, but there is subclass relation it is
    give a non-taxonomic label.
    """
    term1_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]])
    term1_lemma = term1_lemma.replace(' - ', ' ')
    term2_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]])
    term2_lemma = term2_lemma.replace(' - ', ' ')
    term_pair = (term1_lemma, term2_lemma)
    
    if term_pair in relations['subclass-of']:
        return HYPONYM
    elif (term_pair[1], term_pair[0]) in relations['subclass-of']:
        return HYPERNYM
    else:
        return ABSTAIN
    
def _kb_neg(cand, terms, relations):
    
    term1_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]])
    term1_lemma = term1_lemma.replace(' - ', ' ')
    term2_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]])
    term2_lemma = term2_lemma.replace(' - ', ' ')
    term_pair = (term1_lemma, term2_lemma)
    
    if term_pair not in relations['subclass-of'] and \
       (term_pair[1], term_pair[0]) not in relations['subclass-of']:
        if term_pair[0] in terms and term_pair[1] in terms:
            return OTHER
    return ABSTAIN

@labeling_function(resources=dict(relations=relations, terms=terms))
def kb_bio101_ds_negative(cand, terms, relations):
    """
    Looks up term pair KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a HYPONYM/HYPERNYM label depending
    on term pair ordering. If it finds both terms in the KB, but there is subclass relation it is
    give a non-taxonomic label.
    """
    return _kb_neg(cand, terms, relations)

@labeling_function(resources=dict(relations=relations, terms=terms))
def kb_bio101_ds_neg2(cand, terms, relations):
    return _kb_neg(cand, terms, relations)

@labeling_function(resources=dict(relations=relations, terms=terms))
def kb_bio101_ds_neg3(cand, terms, relations):
    return _kb_neg(cand, terms, relations)
    
# labeling functions to apply
label_fns = [
    isa_pattern, 
    suchas_pattern, 
    including_pattern, 
    called_pattern, 
    especially_pattern,
    appo_pattern, 
    other_pattern, 
    are_pattern,
    whichis_pattern,
    term_part_of_speech,
    term_dep_role,
    term_modifier,
    term_subset,
    kb_bio101_ds_taxonomy,
    kb_bio101_ds_negative,
    nsubj_pattern
]
