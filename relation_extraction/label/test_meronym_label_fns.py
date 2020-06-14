import unittest
import pandas as pd
import spacy
from meronym_label_fns import *
from label_constants import *

test_data = [
    { 
        'test_id': 'in_pattern_01',
        'label_fn': in_pattern_lf,
        'label': label_classes.index('PART-OF'),
        'text': "The number of chromosomes in a gamete is 3", 
        'term1': 'chromosome',
        'term1_location': (3, 4),
        'term2': 'gamete',
        'term2_location': (6, 7)
    },
    { 
        'test_id': 'has_pattern_01',
        'label_fn': has_pattern_lf,
        'label': label_classes.index('HAS-PART'),
        'text': "Like other cells, each neuron has a cell body, nucleus, and membrane.", 
        'term1': 'neuron',
        'term1_location': (5, 6),
        'term2': 'nucleus',
        'term2_location': (11, 12)
    },
    { 
        'test_id': 'of_pattern_01',
        'label_fn': of_pattern_lf,
        'label': label_classes.index('PART-OF'),
        'text': "Here is the phospholipid bilayer of the new cell membrane.",
        'term1': 'phospholipid bilayer',
        'term1_location': (3, 5),
        'term2': 'cell membrane',
        'term2_location': (8, 10)
    },
    { 
        'test_id': 'partof_pattern_01',
        'label_fn': partof_pattern_lf,
        'label': label_classes.index('PART-OF'),
        'text': "The hilum is the concave part of the bean." 
        'term1': 'phospholipid bilayer',
        'term1_location': (3, 5),
        'term2': 'cell membrane',
        'term2_location': (8, 10)
    }
    
    
]
class TestMeronymLabelFns(unittest.TestCase):
    
    def setUp(self):
        nlp = spacy.load('en_core_web_sm')
        
        self.test_data = pd.DataFrame(test_data)
        self.test_data['doc'] = [nlp(x.text) for _, x in self.test_data.iterrows()]
    
    def _test_generic(self, test_id):
        row = self.test_data.loc[self.test_data.test_id == test_id, :].squeeze()
        relation = row.label_fn(row)
        self.assertEqual(relation, row.label)
    
    def test_in_pattern_01(self):
        self._test_generic('in_pattern_01')
        
    def test_has_pattern_01(self):
        self._test_generic('has_pattern_01')
        
    def test_of_pattern_01(self):
        self._test_generic('of_pattern_01')
        
if __name__ == '__main__':
    unittest.main()