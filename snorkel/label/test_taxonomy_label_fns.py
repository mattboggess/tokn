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
        'text': "You are probably most familiar with keratin, a fibrous protein that strengthens your hair, nails, and the skin's epidermis.",
        'term1': 'keratin',
        'term1_location': (6, 7),
        'term2': 'fibrous protein',
        'term2_location': (9, 11)
    },
    { 
        'test_id': 'appo_pattern_02', 
        'label_fn': appo_pattern,
        'label': HYPONYM,
        'text': "Cadherin, an elongated protein in the plasma membrane connect to intermediate filaments to create desmosomes.",
        'term1': 'cadherin',
        'term1_location': (0, 1),
        'term2': 'protein',
        'term2_location': (4, 5)
    },
    {
        'test_id': 'appo_pattern_03', 
        'label_fn': appo_pattern,
        'label': ABSTAIN,
        'text': "Further breakdown of food takes place in the small intestine where enzymes produced by the liver, the small intestine, and the pancreas continue the process of digestion.",
        'term1': 'liver',
        'term1_location': (15, 16),
        'term2': 'small intestine',
        'term2_location': (18, 20)
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
    },
    {
        'test_id': 'are_pattern_01', 
        'label_fn': are_pattern,
        'label': HYPONYM,
        'text': "Fats and oils are triglycerides.",
        'term1': 'fat',
        'term1_location': (0, 1),
        'term2': 'triglyceride',
        'term2_location': (4, 5)
    },
    {
        'test_id': 'are_pattern_02', 
        'label_fn': are_pattern,
        'label': HYPONYM,
        'text': "Fats and oils are triglycerides.",
        'term1': 'oil',
        'term1_location': (2, 3),
        'term2': 'triglyceride',
        'term2_location': (4, 5)
    },
    {
        'test_id': 'are_pattern_03', 
        'label_fn': are_pattern,
        'label': HYPERNYM,
        'text': "The receptors are actually sodium channels that open to allow the passage of Na + into the cell when they receive a neurotransmitter signal.",
        'term1': 'receptor',
        'term1_location': (1, 2),
        'term2': 'sodium channel',
        'term2_location': (4, 6)
    },
    {
        'test_id': 'whichis_pattern_01', 
        'label_fn': whichis_pattern,
        'label': HYPONYM,
        'text': "Most cells have a single nucleus, which is usually the most prominent organelle.",
        'term1': 'nucleus',
        'term1_location': (5, 6),
        'term2': 'organelle',
        'term2_location': (13, 14)
    },
    {
        'test_id': 'isa_pattern_01', 
        'label_fn': isa_pattern,
        'label': HYPONYM,
        'text': "So a fatty acid is a molecule with a hydrophilic end and a long hydrophobic tail.",
        'term1': 'fatty acid',
        'term1_location': (2, 4),
        'term2': 'molecule',
        'term2_location': (6, 7)
    },
    {
        'test_id': 'including_pattern_01', 
        'label_fn': including_pattern,
        'label': HYPERNYM,
        'text': "The hypothalamus controls food intake via feedback from blood glucose and hormones, including insulin, leptin, and ghrelin.",
        'term1': 'hormones',
        'term1_location': (11, 12),
        'term2': 'leptin',
        'term2_location': (16, 17)
    },
    {
        'test_id': 'including_pattern_02', 
        'label_fn': including_pattern,
        'label': ABSTAIN,
        'text': "The hypothalamus controls food intake via feedback from blood glucose and hormones, including insulin, leptin, and ghrelin.",
        'term1': 'glucose',
        'term1_location': (9, 10),
        'term2': 'leptin',
        'term2_location': (16, 17)
    },
    {
        'test_id': 'especially_pattern_01', 
        'label_fn': especially_pattern,
        'label': HYPERNYM,
        'text': "These new technologies were powered using fossil fuels, especially coal.", 
        'term1': 'fossil fuel',
        'term1_location': (6, 8),
        'term2': 'coal',
        'term2_location': (10, 11)
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
        
    def test_are_pattern_01(self):
        self._test_generic('are_pattern_01')
        
    def test_are_pattern_02(self):
        self._test_generic('are_pattern_02')
        
    def test_are_pattern_03(self):
        self._test_generic('are_pattern_03')
        
    def test_whichis_pattern_01(self):
        self._test_generic('whichis_pattern_01')
        
    def test_isa_pattern_01(self):
        self._test_generic('isa_pattern_01')
        
    def test_including_pattern_01(self):
        self._test_generic('including_pattern_01')
        
    def test_including_pattern_02(self):
        self._test_generic('including_pattern_02')
        
    def test_especially_pattern_01(self):
        self._test_generic('especially_pattern_01')

if __name__ == '__main__':
    unittest.main()