import unittest
import spacy
import pandas as pd
from data_processing_utils import get_closest_match, tag_terms

tag_terms_test_data = [
    {
        'test_id': 'basic_test',
        'terms': ['cell', 'cell wall', 'biologist'],
        'text': "A biologist will tell you that a cell contains a cell wall.",
        'found_terms': {
            'biologist': {
                'text': ['biologist'], 
                'tokens': [['biologist']], 
                'pos': [['NN']], 
                'dep': [['nsubj']], 
                'indices': [(1, 2)]
            },
            'cell': {
                'text': ['cell'], 
                'tokens': [['cell']],
                'pos': [['NN']], 
                'dep': [['nsubj']], 
                'indices': [(7, 8)]
            },
            'cell wall': {
                'text': ['cell wall'], 
                'tokens': [['cell', 'wall']],
                'pos': [['NN', 'NN']], 
                'dep': [['compound', 'dobj']], 
                'indices': [(10, 12)]}
        },
        'bioes_tags': ['O', 'S', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'B', 'E', 'O']
    },
    {
        'test_id': 'duplicate_term',
        'terms': ['cell', 'cell wall'],
        'text': "A cell may have a cell wall, but not all cells do.",
        'found_terms': {
            'cell': {
                'text': ['cell', 'cells'], 
                'tokens': [['cell'], ['cells']],
                'pos': [['NN'], ['NNS']], 
                'dep': [['nsubj'], ['nsubj']], 
                'indices': [(1, 2), (11, 12)]
            },
            'cell wall': {
                'text': ['cell wall'], 
                'tokens': [['cell', 'wall']], 
                'pos': [['NN', 'NN']], 
                'dep': [['compound', 'dobj']], 
                'indices': [(5, 7)]
            }
        },
        'bioes_tags': ['O', 'S', 'O', 'O', 'O', 'B', 'E', 'O', 'O', 'O', 'O', 'S', 'O', 'O']
        
    },
    {
        'test_id': 'stop_words',
        'terms': ['cell', 'AND'],
        'text': "Cells can have AND.",
        'found_terms': {
            'cell': {
                'text': ['Cells'], 
                'tokens': [['Cells']],
                'pos': [['NNS']], 
                'dep': [['nsubj']], 
                'indices': [(0, 1)]
            }
        },
        'bioes_tags': ['S', 'O', 'O', 'O', 'O']
        
    },
    {
        'test_id': 'heuristic_match',
        'terms': ['alpha tubulin', 'β-tubulin', 'cell membrane'],
        'text': "α-tubulin and beta tubulin are important for cell-membranes.",
        'found_terms': {
            'alpha tubulin': {
                'text': ['α-tubulin'], 
                'tokens': [['α', '-', 'tubulin']],
                'pos': [['LS', 'HYPH', 'NNP']], 
                'dep': [['meta', 'punct', 'nsubj']], 
                'indices': [(0, 3)]
            },
            'β tubulin': {
                'text': ['beta tubulin'], 
                'tokens': [['beta', 'tubulin']],
                'pos': [['NN', 'NNP']], 
                'dep': [['compound', 'conj']], 
                'indices': [(4, 6)]
            },
            'cell membrane': {
                'text': ['cell-membranes'], 
                'tokens': [['cell', '-', 'membranes']],
                'pos': [['NN', 'HYPH', 'NNS']], 
                'dep': [['compound', 'punct', 'pobj']], 
                'indices': [(9, 12)]
            }
        },
        'bioes_tags': ['B', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'I', 'E', 'O']
    },
    {
        'test_id': 'dep_filter',
        'terms': ['collagen fiber', 'proteoglycan', 'carbohydrate', 'protein', 'molecule'],
        'text': "Collagen fibers are interwoven with proteoglycans, which are carbohydrate-containing protein molecules.",
        'found_terms': {
            'collagen fiber': {
                'text': ['Collagen fibers'], 
                'tokens': [['Collagen', 'fibers']],
                'pos': [['NNP', 'NNS']], 
                'dep': [['compound', 'nsubjpass']], 
                'indices': [(0, 2)]
            },
            'proteoglycan': {
                'text': ['proteoglycans'], 
                'tokens': [['proteoglycans']],
                'pos': [['NNS']], 
                'dep': [['pobj']], 
                'indices': [(5, 6)]
            },
            'molecule': {
                'text': ['molecules'], 
                'tokens': [['molecules']],
                'pos': [['NNS']], 
                'dep': [['attr']], 
                'indices': [(13, 14)]
            }
        },
        'bioes_tags': ['B', 'E', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O']
    },
    {
        'test_id': 'pos_filter',
        'terms': ['cytokinesis', 'divide', 'cytoplasm', 'cell'],
        'text': "Cytokinesis divides the cell's cytoplasm.",
        'found_terms': {
            'cytokinesis': {
                'text': ['Cytokinesis'], 
                'tokens': [['Cytokinesis']],
                'pos': [['NN']], 
                'dep': [['nsubj']], 
                'indices': [(0, 1)]
            },
            'cytoplasm': {
                'text': ['cytoplasm'], 
                'tokens': [['cytoplasm']],
                'pos': [['NN']], 
                'dep': [['dobj']], 
                'indices': [(5, 6)]
            }
        },
        'bioes_tags': ['S', 'O', 'O', 'O', 'O', 'S', 'O'] 
    },
    {
        'test_id': 'dep_filter2',
        'terms': ['biological macromolecule', 'nitrogen'],
        'text': "This exoskeleton is made of the biological macromolecule chitin , which is a polysaccharide-containing nitrogen.",
        'found_terms': {
            'nitrogen': {
                'text': ['nitrogen'], 
                'tokens': [['nitrogen']],
                'pos': [['NN']], 
                'dep': [['attr']], 
                'indices': [(16, 17)]
            }
        },
        'bioes_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O'] 
    },
    {
        'test_id': 'pos_filter2',
        'terms': ['regulate'],
        'text': "Metabolic pathways are regulated systems Figure 9.13 Relationships among the Major Metabolic Pathways of the Cell",
        'found_terms' : {},
        'bioes_tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    }
    
]

class TestDataProcessingUtils(unittest.TestCase):
    
    def setUp(self):
        self.test_data = pd.DataFrame(tag_terms_test_data)
    
    def _test_tag_terms_generic(self, test_id, invalid_dep=[], invalid_pos=[]):
        row = self.test_data.loc[self.test_data.test_id == test_id, :].squeeze()
        output = tag_terms(row.text, row.terms, invalid_dep=invalid_dep, invalid_pos=invalid_pos)
        self.assertEqual(output['tags'], row.bioes_tags)
        self.assertDictEqual(output['found_terms'], row.found_terms)
    
    def test_get_closest_matches(self):
        
        indices1 = [(12, 15)]
        indices2 = [(18, 20)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[0], indices2[0]))
        
        indices1 = [(12, 15), (5, 6)]
        indices2 = [(8, 10)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[1], indices2[0]))
    
    def test_tag_terms_basic(self):
        self._test_tag_terms_generic('basic_test')
        
    def test_tag_terms_duplicate_term(self):
        self._test_tag_terms_generic('duplicate_term')
        
    def test_tag_terms_stop_words(self):
        self._test_tag_terms_generic('stop_words')
        
    def test_heuristic_matches(self):
        self._test_tag_terms_generic('heuristic_match')
        
    def test_dep_filter(self):
        self._test_tag_terms_generic('dep_filter', invalid_dep=['npadvmod', 'compound', 'admod'])
        
    def test_dep_filter2(self):
        self._test_tag_terms_generic('dep_filter2', invalid_dep=['npadvmod', 'compound', 'admod'])
        
    #def test_uncommon_plural_match(self):
    #    self._test_tag_terms_generic('uncommon_plural')
        
    def test_pos_filter(self):
        self._test_tag_terms_generic('pos_filter', invalid_dep=['poss'], invalid_pos=['VBZ'])
        
    def test_pos_filter2(self):
        self._test_tag_terms_generic('pos_filter', invalid_dep=['poss'], invalid_pos=['VBZ'])

        
if __name__ == '__main__':
    unittest.main()