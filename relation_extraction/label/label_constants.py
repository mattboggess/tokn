# Important mappings, lists, and constants shared across labeling functions

ABSTAIN = -1
taxonomy_classes = [
    'SUBCLASS',
    'SUPERCLASS',
    'SYNONYM'
]
meronym_classes = [
    'HAS-PART/REGION',
    'PART/REGION-OF',
]

label_classes = ['OTHER'] + taxonomy_classes + meronym_classes

kb_bio101_mapping = {
    'subclass-of': [label_classes.index('SUBCLASS'), 
                    label_classes.index('SUPERCLASS')],
    'synonym': [label_classes.index('SYNONYM'), 
                label_classes.index('SYNONYM')],
    'has-part': [label_classes.index('HAS-PART/REGION'), 
                 label_classes.index('PART/REGION-OF')],
    'has-region': [label_classes.index('HAS-PART/REGION'), 
                   label_classes.index('PART/REGION-OF')]
}

def human_label(x, label_classes):
    if x == -1:
        return 'ABSTAIN'
    else:
        return label_classes[x]