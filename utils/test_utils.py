import unittest
from utils import tag_text, tag_relations
import spacy


class TestUtils(unittest.TestCase):

    def test_tag_text(self):

        terms = ['cell', 'cell wall', 'biologist'] 
        text = 'A biologist will tell you that a cell contains a cell wall.'
        relations = [('cell', 'has-part', 'cell wall')]

        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [1]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [7]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [10]}}
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        bioes_tags = ["O", "S", "O", "O", "O", "O", "O", "S", "O", "O", "B", "E", "O"]
        found_relations = [("biologist", "no-relation", "cell"),
                           ("biologist", "no-relation", "cell wall"),
                           ("cell", "has-part", "cell wall")]
        solution = {"input_text": text, "tokenized_text": tokenized_text, "tagged_text": bioes_tags,
                    "relations": found_relations, "terms": found_terms}
        output = tag_text(text, terms, relations)

        self.assertEqual(output, solution)
    
    def test_tag_relations(self):
        
        terms = ["cell", "cell wall", "virus"]
        relations = [("cell", "has-part", "cell wall"),
                     ("cell", "test-double", "cell wall"),
                     ("cell", "no-relation", "virus"),
                     ("cell wall", "no-relation", "virus")]
        found_relations = tag_relations(terms, relations)
        self.assertEqual(set(relations), set(found_relations))


if __name__ == "__main__":
    unittest.main()