import unittest
import spacy
from data_processing import get_closest_match, tag_terms, insert_relation_tags, add_relation


class TestUtils(unittest.TestCase):
    
    def test_get_closet_matches(self):
        
        indices1 = [(12, 15)]
        indices2 = [(18, 20)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[0], indices2[0]))
        
        indices1 = [(12, 15), (5, 6)]
        indices2 = [(8, 10)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[1], indices2[0]))

    def test_tag_terms(self):

        terms = ['cell', 'cell wall', 'biologist'] 
        text = 'A biologist will tell you that a cell contains a cell wall.'

        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [(1, 2)]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [(7, 8)]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [(10, 12)]}}
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        bioes_tags = ["O", "S", "O", "O", "O", "O", "O", "S", "O", "O", "B", "E", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output, solution)
       
    def test_insert_relation_tags(self):
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        indices = ((1, 2), (10, 12))
        solution = ["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", "a", "cell", 
                    "contains", "a", "<e2>", "cell", "wall", "</e2>", "."]
        output = insert_relation_tags(tokenized_text, indices)
        self.assertEqual(output, solution)
    
    def test_add_relation(self):
        
        # inputs
        relations_db = {
            "has-part": {
                "cell -> cell wall": {
                    "sentences": [],
                    "e1_representations": ["cell"],
                    "e2_representations": ["cell wall"],
                }
            },
            "no-relation": {}
        }
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [(1, 2)]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [(7, 8)]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [(10, 12)]}}
        
        # solution
        haspart_sentence = ["A", "biologist", "will", "tell", "you", "that", "a", "<e1>", "cell", 
                            "</e1>", "contains", "a", "<e2>", "cell", "wall", "</e2>", "."]
        nr_sent = ["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", 
                    "a", "<e2>", "cell", "</e2>", "contains", "a", "cell", "wall", "."]
        relations_db_solution = {
            "has-part": {
                "cell -> cell wall": {
                    "sentences": [haspart_sentence],
                    "e1_representations": ["cell"],
                    "e2_representations": ["cell wall"],
                }
            },
            "no-relation": {
                "biologist -> cell": {
                    "sentences": [nr_sent],
                    "e1_representations": ["biologist"],
                    "e2_representations": ["cell"]
                }
            }
        }
        
        output = add_relation(("biologist", "cell"), found_terms, tokenized_text, 
                              relations_db)
        self.assertEqual(output["no-relation"], relations_db_solution["no-relation"])
  
#    def test_tag_relations(self):
        
#         terms = ["cell", "cell wall", "virus"]
#         relations = [("cell", "has-part", "cell wall"),
#                      ("cell", "test-double", "cell wall"),
#                      ("cell", "no-relation", "virus"),
#                      ("cell wall", "no-relation", "virus")]
#         found_relations = tag_relations(terms, relations)
#         self.assertEqual(set(relations), set(found_relations))


if __name__ == "__main__":
    unittest.main()