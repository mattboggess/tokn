from label_constants import *
from snorkel.labeling import labeling_function
import pickle

with open("../data/kb_bio101_relations_db.pkl", 'rb') as fid:
    kb_bio101 = pickle.load(fid)
with open("../data/kb_bio101_terms.pkl", 'rb') as fid:
    kb_terms = pickle.load(fid)

# ==============================================================
# Positive Distant Supervision

def _kb_bio101_ds_positive(cand, kb_bio101, kb_relation, relation_labels):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    term1_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]])
    term1_lemma = term1_lemma.replace(' - ', ' ')
    term2_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]])
    term2_lemma = term2_lemma.replace(' - ', ' ')
    term_pair = (term1_lemma, term2_lemma)
    
    if term_pair in kb_bio101[kb_relation]:
        return relation_labels[0]
    elif (term_pair[1], term_pair[0]) in kb_bio101[kb_relation]:
        return relation_labels[1] 
    else:
        return ABSTAIN

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_taxonomy(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    return _kb_bio101_ds_positive(cand, kb_bio101, 'subclass-of', kb_bio101_mapping['subclass-of'])

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_synonym(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    if _kb_bio101_ds_positive(cand, kb_bio101, 'subclass-of', kb_bio101_mapping['subclass-of']) != ABSTAIN:
        return ABSTAIN
    return _kb_bio101_ds_positive(cand, kb_bio101, 'synonym', kb_bio101_mapping['synonym'])

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_has_part(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    return _kb_bio101_ds_positive(cand, kb_bio101, 'has-part', kb_bio101_mapping['has-part'])

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_has_region(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    return _kb_bio101_ds_positive(cand, kb_bio101, 'has-region', kb_bio101_mapping['has-region'])

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_possesses(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    return _kb_bio101_ds_positive(cand, kb_bio101, 'possesses', kb_bio101_mapping['possesses'])

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_element(cand, kb_bio101, kb_bio101_mapping):
    """
    Looks up term pair in KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. 
    """
    return _kb_bio101_ds_positive(cand, kb_bio101, 'element', kb_bio101_mapping['element'])
    
# ==============================================================
# Negative Distant Supervision
    
def _kb_bio101_ds_negative(cand, kb_bio101, kb_terms, kb_bio101_mapping):
    
    term1_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term1_location[0]:cand.term1_location[1]]])
    term1_lemma = term1_lemma.replace(' - ', ' ')
    term2_lemma = ' '.join([tok.lemma_ 
                            for tok in cand.doc[cand.term2_location[0]:cand.term2_location[1]]])
    term2_lemma = term2_lemma.replace(' - ', ' ')
    term_pair = (term1_lemma, term2_lemma)
    
    # abstain if there is some positive relation
    for kb_relation, relation_labels in kb_bio101_mapping.items():
        if _kb_bio101_ds_positive(cand, kb_bio101, kb_relation, relation_labels) != -1:
            return ABSTAIN
    
    # return other if individual terms are present in KB
    if term_pair[0] in kb_terms and term_pair[1] in kb_terms:
        return label_classes.index('OTHER')
    
    return ABSTAIN

@labeling_function(resources=dict(kb_bio101=kb_bio101, kb_terms=kb_terms, kb_bio101_mapping=kb_bio101_mapping))
def kb_bio101_ds_negative(cand, kb_bio101, kb_terms, kb_bio101_mapping):
    """
    Looks up term pair KB Bio101 knowledge base manually built on the first 10 chapters of Life
    Biology. If it finds a subclass relation there it provides a SUBCLASS/SUPERCLASS label depending
    on term pair ordering. If it finds both terms in the KB, but there is subclass relation it is
    give a non-taxonomic label.
    """
    return _kb_bio101_ds_negative(cand, kb_bio101, kb_terms, kb_bio101_mapping)

taxonomy_kb_fns = [
    kb_bio101_ds_taxonomy,
    kb_bio101_ds_synonym
]

meronym_kb_fns = [
    kb_bio101_ds_has_part,
    kb_bio101_ds_has_region
    #kb_bio101_ds_possesses,
    #kb_bio101_ds_element
]