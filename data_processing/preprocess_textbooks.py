import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
import warnings
import pandas as pd
import re

OPENSTAX_TEXTBOOKS = ["Anatomy_and_Physiology", "Astronomy", "Biology_2e", "Chemistry_2e",
                      "Microbiology", "Psychology", "University_Physics_Volume_1",
                      "University_Physics_Volume_2", "University_Physics_Volume_3"]

EXCLUDE_SECTIONS = [
    "Preface", "Chapter Outline", "Index", "Chapter Outline", "Summary", "Multiple Choice",
    "Fill in the Blank", "short Answer", "Critical Thinking", "References", 
    "Units", "Conversion Factors", "Fundamental Constants", "Astronomical Data",
    "Mathematical Formulas", "The Greek Alphabet", "Chapter 1", "Chapter 2",
    "Chapter 3", "Chapter 4", "Chapter 5", "Chapter 6", "Chapter 7", "Chapter 8"
    "Chapter 9", "Chapter 10", "Chapter 11", "Chapter 12", "Chapter 13", "Chapter 14", 
    "Chapter 15", "Chapter 16", "Chapter 17", "Critical Thinking Questions", 
    "Visual Connection Questions", "Key Terms", "Review Questions", 
    "The Periodic Table of Elements", "Measurements and the Metric System"]

def process_term(key_term_text):
    """ Parse openstax data to extract key terms (including acronyms). """
    if ":" not in key_term_text:
        return []
    term = key_term_text.split(":")[0]
    match = re.match(".*\((.+)\).*", term)
    if match:
        acronym = match.group(1)
        term = term.replace(f"(acronym)", "")
        return [term.strip(), acronym.strip()]
    
    return [term.strip()] 

if __name__ == "__main__":
    
    raw_data_dir = "../data/raw_data"
    spacy_data_dir = "../data/spacy_preprocessed"
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    # process openstax textbooks
    all_terms_spacy = []
    all_terms_text = [] 
    for textbook in OPENSTAX_TEXTBOOKS:
        print(f"Processing {textbook} textbook")
        textbook_data = pd.read_csv(f"{raw_data_dir}/textbooks/sentences_{textbook}_parsed.csv")
        
        # spacy preprocess sentences
        output_file = f"{spacy_data_dir}/{textbook}_sentences_spacy"
        sentences = textbook_data[~textbook_data.section_name.isin(EXCLUDE_SECTIONS)].sentence
        sentences_spacy = []
        for i, sent in enumerate(sentences):
            if i % 500 == 0:
                print(f"Processing sentence {i}/{len(sentences)}")
            if i == 1000:
                break
            sentences_spacy.append(nlp(sent))
        write_spacy_docs(sentences_spacy, output_file)
        
        # spacy preprocess terms
        output_file = f"{spacy_data_dir}/{textbook}_key_terms_spacy"
        key_terms = textbook_data[textbook_data.section_name == "Key Terms"].sentence
        key_terms_spacy = []
        for i, key_term in enumerate(key_terms):
            if i % 500 == 0:
                print(f"Processing Key Term {i}/{len(key_terms)}")
            kts = process_term(key_term)
            if len(kts):
                key_terms_spacy += [nlp(kt) for kt in kts]

        write_spacy_docs(key_terms_spacy, output_file)
        
        # compile full list of key terms w/o duplicates
        all_terms_spacy += [term for term in key_terms_spacy if term.text not in all_terms_text] 
        all_terms_text += [term.text for term in key_terms_spacy]
        
    # TODO: Process life biology textbook
    
    write_spacy_docs(all_terms_spacy, f"{spacy_data_dir}/all_key_terms_spacy")
    