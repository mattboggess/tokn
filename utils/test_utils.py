import unittest
from utils import tag_terms
import spacy


class TestUtils(unittest.TestCase):

    def test_tag_terms(self):

        terms = ["cell", "organism", "cell theory", "neuron"]

        text = ("They formulated their conclusion as the cell theory, which states that: "
                "Cells are the basic structural and physiological units of all living organisms.")

        found_terms_solution = [("cell theory", [6]), ("organism", [24]), ("cell", [13])]
        tokenized_text_solution = ["They", "formulated", "their", "conclusion", "as", "the", "cell",
                                   "theory", ",", "which", "states", "that", ":", "Cells", "are", "the",
                                   "basic", "structural", "and", "physiological", "units", "of", "all",
                                   "living", "organisms", "."]
        tags_solution = ["O", "O", "O", "O", "O", "O", "B", "E", "O", "O", "O", "O", "O",
                         "S", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "S", "O"]

        found_terms, tokenized_text, tags = tag_terms(text, terms)
        self.assertEqual(found_terms, found_terms_solution)
        self.assertEqual(tokenized_text, tokenized_text_solution)
        self.assertEqual(tags, tags_solution)


if __name__ == "__main__":
    unittest.main()