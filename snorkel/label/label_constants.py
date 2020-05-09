

ABSTAIN = -1
OTHER = 0
HYPONYM = 1
HYPERNYM = 2
SYNONYM = 3

label_classes = ['OTHER', 'HYPONYM', 'HYPERNYM', 'SYNONYM']

kb_bio101_mapping = {
    'subclass-of': [HYPONYM, HYPERNYM],
    'synonym': [SYNONYM, SYNONYM]
}