import unittest
import spacy
from data_processing_utils import get_closest_match, tag_terms, insert_relation_tags, add_relation
from data_processing_utils import read_spacy_docs

class TestDataProcessingUtils(unittest.TestCase):
    
    def test_get_closest_matches(self):
        
        indices1 = [(12, 15)]
        indices2 = [(18, 20)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[0], indices2[0]))
        
        indices1 = [(12, 15), (5, 6)]
        indices2 = [(8, 10)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[1], indices2[0]))

    def test_tag_terms(self):
        terms = read_spacy_docs("../data/preprocessed_data/Life_Biology_kb_key_terms_spacy") 
        text = 'By the end of this section, you will be able to do the following:    Describe the structure of eukaryotic cells   Compare animal cells with plant cells   State the role of the plasma membrane   Summarize the functions of the major cell organelles       Have you ever heard the phrase "form follows function?"'

        terms = ['cell', 'cell wall', 'biologist'] 
        text = 'A biologist will tell you that a cell contains a cell wall.'

        found_terms = {"biologist": {"text": ["biologist"], "pos": ["NN"], "indices": [(1, 2)], "type": ["entity"]},
                       "cell": {"text": ["cell"], "pos": ["NN"], "indices": [(7, 8)], "type": ["entity"]},
                       "cell wall": {"text": ["cell wall"], "pos": ["NN NN"], "indices": [(10, 12)], "type": ["entity"]}}
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        bioes_tags = ["O", "S", "O", "O", "O", "O", "O", "S", "O", "O", "B", "E", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        terms = ['pineapple plant', 'renal-disease', 'zinc atom'] 
        text = 'Here are some troublesome terms: pineapples, renal disease, zinc, and more.'

        found_terms = {"pineapple plant": {"text": ["pineapples"], "pos": ["NNS"], "indices": [(6, 7)], "type": ["entity"]},
                       "renal-disease": {"text": ["renal disease"], "pos": ["JJ NN"], 
                                         "indices": [(8, 10)], "type": ["entity"]},
                       "zinc atom": {"text": ["zinc"], "pos": ["NN"], "indices": [(11, 12)], "type": ["entity"]}}
        tokenized_text = ["Here", "are", "some", "troublesome", "terms", ":", "pineapples", ",",
                          "renal", "disease", ",", "zinc", ",", "and", "more", "."]
        bioes_tags = ["O", "O", "O", "O", "O", "O", "S", "O", "B", "E", "O", "S", "O", "O", "O", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        # test case insensitive matching
        text = 'Acid reflux or heartburn occurs'
        terms = ['acid reflux'] 
        found_terms = {"acid reflux": {"text": ["Acid reflux"], "pos": ["NN NN"], 
                                       "indices": [(0, 2)], "type": ["entity"]}}
        tokenized_text = ["Acid", "reflux", "or", "heartburn", "occurs"]
        bioes_tags = ["B", "E", "O", "O", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        text = "Figure 12.6 Alkaptonuria is a recessive genetic disorder"
        terms = ['alkaptonuria'] 
        found_terms = {"alkaptonuria": {"text": ["Alkaptonuria"], "pos": ["NNP"], 
                                       "indices": [(2, 3)], "type": ["entity"]}}
        tokenized_text = ["Figure", "12.6", "Alkaptonuria", "is", "a", "recessive", "genetic", "disorder"]
        bioes_tags = ["O", "O", "S", "O", "O", "O", "O", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        text = "In anaphase II, the sister chromatids separate."
        terms = ['anaphase ii'] 
        found_terms = {"anaphase ii": {"text": ["anaphase II"], "pos": ["NN CD"], 
                                       "indices": [(1, 3)], "type": ["entity"]}}
        tokenized_text = ["In", "anaphase", "II", ",", "the", "sister", "chromatids", "separate", "."]
        bioes_tags = ["O", "B", "E", "O", "O", "O", "O", "O", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        # test special character replacing
        text = "the 3′ hydroxyl group and α tubulin exist"
        terms = ['3 prime hydroxyl group', 'alpha tubulin'] 
        found_terms = {"3 prime hydroxyl group": {"text": ["3 ′ hydroxyl group"], 
                                                  "pos": ["CD NN NN NN"], 
                                                  "indices": [(1, 5)], "type": ["entity"]},
                       "alpha tubulin": {"text": ["α tubulin"], "pos": ["JJ NN"], 
                                         "indices": [(6, 8)], "type": ["entity"]}}
        tokenized_text = ["the", "3", "′", "hydroxyl", "group", "and", "α", "tubulin", "exist"] 
        bioes_tags = ["O", "B", "I", "I", "E", "O", "B", "E", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output["tokenized_text"], solution[0])
        self.assertEqual(output["tags"], solution[1])
        self.assertDictEqual(output["found_terms"], solution[2])
        
        
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
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "cells", "contain",
                          "a", "cell", "wall", "."]
        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [(1, 2)]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [(6, 7)]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [(9, 11)]}}
        
        # solution
        haspart_sentence = " ".join(["A", "biologist", "will", "tell", "you", "that",  "<e1>", 
                                     "cells", "</e1>", "contain", "a", "<e2>", "cell", "wall", 
                                     "</e2>", "."])
        nr_sent = " ".join(["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", 
                            "<e2>", "cells", "</e2>", "contain", "a", "cell", "wall", "."])
        relations_db_solution = {
            "has-part": {
                "cell -> cell wall": {
                    "sentences": [haspart_sentence],
                    "e1_representations": ["cell", "cells"],
                    "e2_representations": ["cell wall"],
                }
            },
            "no-relation": {
                "biologist -> cell": {
                    "sentences": [nr_sent],
                    "e1_representations": ["biologist"],
                    "e2_representations": ["cells"]
                }
            }
        }
        
        output = add_relation(("cell", "cell wall"), found_terms, tokenized_text, 
                              relations_db)
        self.assertEqual(output["has-part"], relations_db_solution["has-part"])
        output = add_relation(("biologist", "cell"), found_terms, tokenized_text, 
                              relations_db)
        self.assertEqual(output["no-relation"], relations_db_solution["no-relation"])
  
if __name__ == "__main__":
    unittest.main()