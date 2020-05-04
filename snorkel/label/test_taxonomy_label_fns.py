import unittest
import pandas as pd
import spacy
from taxonomy_label_fns import *

test_data = [
    { 
        'test_id': 'other_pattern_01',
        'label_fn': other_pattern,
        'label': HYPONYM,
        'text': "They bear tension, thus maintaining the cell's shape, and anchor the nucleus and other organelles in place.",
        'term1': 'nucleus',
        'term1_location': (14, 15),
        'term2': 'organelle',
        'term2_location': (17, 18)
    },
    { 
        'test_id': 'appo_pattern_01', 
        'label_fn': appo_pattern,
        'label': HYPONYM,
        'text': "You are probably most familiar with keratin, the fibrous protein that strengthens your hair, nails, and the skin's epidermis.",
        'term1': 'keratin',
        'term1_location': (6, 7),
        'term2': 'fibrous protein',
        'term2_location': (9, 11)
    },
    { 
        'test_id': 'appo_pattern_02', 
        'label_fn': appo_pattern,
        'label': HYPONYM,
        'text': "Cadherins, short proteins in the plasma membrane connect to intermediate filaments to create desmosomes.",
        'term1': 'cadherin',
        'term1_location': (0, 1),
        'term2': 'protein',
        'term2_location': (3, 4)
    },
    {
        'test_id': 'appo_pattern_03', 
        'label_fn': appo_pattern,
        'label': OTHER,
        'text': "Figure 10.12 Cyclin-dependent kinases (Cdks) are protein kinases that, when fully activated, can phosphorylate and thus activate other proteins that advance the cell cycle past a checkpoint.",
        'term1': 'cyclin dependent kinase',
        'term1_location': (3, 6),
        'term2': 'cdk',
        'term2_location': (7, 8)
    },
    {
        'test_id': 'appo_pattern_04', 
        'label_fn': appo_pattern,
        'label': ABSTAIN,
        'text': "Lysosomes digest macromolecules, recycle worn-out organelles, and destroy pathogens.",
        'term1': 'lysosome',
        'term1_location': (0, 1),
        'term2': 'macromolecule',
        'term2_location': (2, 3)
    },
    { 
        'test_id': 'called_pattern_01', 
        'label_fn': called_pattern,
        'label': HYPERNYM,
        'text': "In the active , dephosphorylated state, Rb binds to proteins called transcription factors , most commonly, E2F ( Figure 10.13 ).",
        'term1': 'protein',
        'term1_location': (10, 11),
        'term2': 'transcription factor',
        'term2_location': (12, 14)
    },
    {
        'test_id': 'called_pattern_02', 
        'label_fn': called_pattern,
        'label': OTHER,
        'text': "The spindle apparatus (also called the mitotic spindle or simply the spindle) is a dynamic microtubule structure that moves sister chromatids apart during mitosis.",
        'term1': 'spindle apparatus',
        'term1_location': (1, 3),
        'term2': 'mitotic spindle',
        'term2_location': (7, 9)
    }
]

class TestTaxonomyLabelFns(unittest.TestCase):
    
    def setUp(self):
        nlp = spacy.load('en_core_web_sm')
        
        self.test_data = pd.DataFrame(test_data)
        self.test_data['doc'] = [nlp(x.text) for _, x in self.test_data.iterrows()]
    
    def _test_generic(self, test_id):
        row = self.test_data.loc[self.test_data.test_id == test_id, :].squeeze()
        relation = row.label_fn(row)
        self.assertEqual(relation, row.label)
    
    def test_other_pattern_01(self):
        self._test_generic('other_pattern_01')
        
    def test_appo_pattern_01(self):
        self._test_generic('appo_pattern_01')
        
    def test_appo_pattern_02(self):
        self._test_generic('appo_pattern_02')
        
    def test_appo_pattern_03(self):
        self._test_generic('appo_pattern_03')
        
    def test_appo_pattern_04(self):
        self._test_generic('appo_pattern_04')
        
    def test_called_pattern_01(self):
        self._test_generic('called_pattern_01')
        
    def test_called_pattern_02(self):
        self._test_generic('called_pattern_02')

if __name__ == '__main__':
    unittest.main()