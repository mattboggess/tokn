##########
# Preprocesses Openstax and Life Biology textbooks using Spacy and saves to disk. 
# Creates separate sentence and key term files for each.
# For Life, does additional parsing to extract out the knowledge base lexicon and first 10 chapters.
##########
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
from data_processing_constants import OPENSTAX_TEXTBOOKS, EXCLUDE_SECTIONS
import warnings
from io import StringIO
import pandas as pd
import re
import json

def parse_openstax_terms(key_term_text):
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

def process_lexicon(lexicon):
    """ Takes in a lexicon consisting of concept text representation pairs and turns this into a 
    list of Spacy processed terms and a lexicon csv mapping KB concepts to lists of text 
    representations and their lemma forms.
    """
    
    # get rid of extra column and read in as dataframe
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, 
                          names=["concept", "relation", "text", "pos"])

    # create mapping from kb concept to unique text representations
    lexicon = lexicon[~lexicon.text.str.contains("Concept-Word-Frame")]
    lexicon = lexicon.groupby("concept")["text"].apply(lambda x: list(set(x))).reset_index()

    # spacy process terms to get lemmas
    spacy_terms = []
    lemmas = []
    for concept in lexicon.concept:
        terms = list(lexicon.loc[lexicon.concept == concept, "text"])[0]
        terms = [t.replace('"', "").strip() for t in terms]
        spacy_terms_tmp = [nlp(term) for term in terms]
        lemma_terms = list([" ".join([tok.lemma_ for tok in t]) for t in spacy_terms_tmp])
        spacy_terms += spacy_terms_tmp
        lemmas.append(lemma_terms)

    lexicon["lemmas"] = lemmas
    return spacy_terms, lexicon

if __name__ == "__main__":
    
    raw_data_dir = "../data/raw_data"
    preprocessed_data_dir = "../data/preprocessed_data"
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    # process openstax textbooks
    for textbook in OPENSTAX_TEXTBOOKS:
        continue
        print(f"Processing {textbook} textbook")
        textbook_data = pd.read_csv(f"{raw_data_dir}/openstax/sentences_{textbook}_parsed.csv")
        
        # spacy preprocess sentences
        output_file = f"{preprocessed_data_dir}/{textbook}_sentences_spacy"
        sentences = textbook_data[~textbook_data.section_name.isin(EXCLUDE_SECTIONS)].sentence
        sentences_spacy = []
        for i, sent in enumerate(sentences):
            if i % 500 == 0:
                print(f"Processing sentence {i}/{len(sentences)}")
            if not len(sent):
                continue
            sentences_spacy.append(nlp(sent))
        write_spacy_docs(sentences_spacy, output_file)
        
        # spacy preprocess terms
        output_file = f"{preprocessed_data_dir}/{textbook}_key_terms_spacy"
        key_terms_spacy = []
        # use special extracted glossary for microbiology since key term format is different
        if textbook == "Microbiology":
            with open(f"{raw_data_dir}/openstax/microbiology_glossary.json", "r") as f:
                key_terms = json.load(f)
            for i, key_term in enumerate(key_terms):
                if i % 500 == 0:
                    print(f"Processing Key Term {i}/{len(key_terms)}")
                key_terms_spacy.append(nlp(key_term))
                acronym = key_terms[key_term]["acronym"]
                if len(acronym):
                    key_terms_spacy.append(nlp(acronym))
        else:
            key_terms = textbook_data[textbook_data.section_name == "Key Terms"].sentence
            for i, key_term in enumerate(key_terms):
                if i % 500 == 0:
                    print(f"Processing Key Term {i}/{len(key_terms)}")
                kts = parse_openstax_terms(key_term)
                if len(kts):
                    key_terms_spacy += [nlp(kt) for kt in kts]

        write_spacy_docs(key_terms_spacy, output_file)
        
    print("Processing Life Biology Lexicon")
    lexicon_input_file = f"{raw_data_dir}/life_bio/kb_lexicon.txt"
    lexicon_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_lexicon.csv"
    terms_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_terms_spacy"
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    terms, lexicon = process_lexicon(lexicon)
    write_spacy_docs(terms, terms_output_file)
    lexicon.to_csv(lexicon_output_file, index=False)
    
    print("Processing Life Biology Sentences")
    #TODO: Get access to entire Life Biology textbook
    life_input_file = f"{raw_data_dir}/life_bio/life_bio_selected_sentences.txt"
    life_kb_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_sentences_spacy"
    life_output_file = f"{preprocessed_data_dir}/Life_Biology_sentences_spacy"
    with open(life_input_file, "r") as f:
        life_bio_sentences = f.readlines()
        
    sentences_kb_spacy = []
    sentences_spacy = []
    for i, sent in enumerate(life_bio_sentences):
        if i % 500 == 0:
            print(f"Preprocessing life biology sentence {i}/{len(life_bio_sentences)}")
            
        spacy_sent = nlp(re.sub("^(\d*\.*)+\s*", "", sent))
        if int(sent.split(".")[1]) <= 10:
            sentences_kb_spacy.append(spacy_sent)
        sentences_spacy.append(spacy_sent)
        
    write_spacy_docs(sentences_spacy, life_output_file)
    write_spacy_docs(sentences_kb_spacy, life_kb_output_file)
    