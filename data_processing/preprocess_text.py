##########
# Preprocesses Openstax and Life Biology textbooks using Spacy and saves to disk. 
# Creates separate sentence and key term files for each.
# For Life, does additional parsing to extract out the knowledge base lexicon and first 10 chapters.
##########
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from data_processing_utils import write_spacy_docs
import warnings
from io import StringIO
import pandas as pd
import re
import json

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
    "Visual Connection Questions", "Key Terms", "Review Questions", "Glossary",
    "The Periodic Table of Elements", "Measurements and the Metric System"]

STOP_WORDS = ["object", "aggregate", "group", "thing", "region"]

def parse_openstax_terms(key_term_text):
    """ Parse openstax data to extract key terms (including acronyms). """
    if ":" not in key_term_text:
        return []
    term = key_term_text.split(":")[0]
    match = re.match(".*\((.+)\).*", term)
    if match:
        acronym = match.group(1)
        term = term.replace(f"({acronym})", "")
        return [term.strip(), acronym.strip()]
    
    return [term.strip()] 

def process_lexicon(lexicon, bio_concepts):
    """ Takes in a lexicon consisting of concept text representation pairs and turns this into a 
    list of Spacy processed terms and a lexicon csv mapping KB concepts to lists of text 
    representations and their lemma forms.
    """
    
    # get rid of extra column and read in as dataframe
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, 
                          names=["concept", "relation", "text", "pos"])
    
    concept_types = lexicon.query("text in ['Entity', 'Event']")
    lexicon = lexicon.query("text not in ['Entity', 'Event']")

    # create mapping from kb concept to unique text representations
    lexicon = lexicon[~lexicon.text.str.contains("Concept-Word-Frame")]
    lexicon = lexicon.groupby("concept")["text"].apply(
        lambda x: list(set([t for t in x if t not in STOP_WORDS]))).reset_index()
    
    # filter out too general upper ontology words, relation concepts, and concepts that only have stop words 
    lexicon = lexicon[lexicon.concept.isin(bio_concepts)]
    lexicon = lexicon[lexicon.text.map(len) > 0]
    lexicon = lexicon[lexicon.text.apply(lambda x: "Relation" not in x)]

    # spacy process terms to get lemmas
    spacy_terms = []
    lexicon_output = {}
    for concept in lexicon.concept:
        
        # extract text representations for the concept
        terms = list(lexicon.loc[lexicon.concept == concept, "text"])[0]
        terms = [t.replace('"', "").strip() for t in terms]
        
        # add text of concept itself
        concept_text = concept.lower().replace("-", " ")
        if concept_text not in terms:
            terms.append(concept_text)
        
        # spacy preprocess and add lemmas
        spacy_terms_tmp = [nlp(term) for term in terms]
        lemma_terms = [" ".join([tok.lemma_ for tok in t]) for t in spacy_terms_tmp]
        
        # accumulate spacy term representations
        spacy_terms += spacy_terms_tmp
        
        # add concept entry to output lexicon
        lexicon_output[concept] = {"text_representations": terms,
                                   "class_label": concept_types.loc[concept_types.concept == concept, "text"].iat[0],
                                   "lemma_representations": list(set(lemma_terms))}

    # filter out upper ontology concepts
    return spacy_terms, lexicon_output

if __name__ == "__main__":
    
    raw_data_dir = "../data/raw_data"
    preprocessed_data_dir = "../data/preprocessed_data"
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    # process openstax textbooks
    for textbook in OPENSTAX_TEXTBOOKS:
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
    bio_concepts_file = f"{raw_data_dir}/life_bio/kb_biology_concepts.txt"
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    with open(bio_concepts_file, "r") as f:
        bio_concepts = set([t.strip() for t in f.readlines()])
        
    terms, lexicon = process_lexicon(lexicon, bio_concepts)
    
    lexicon_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_lexicon.json"
    terms_output_file = f"{preprocessed_data_dir}/Life_Biology_kb_key_terms_spacy"
    write_spacy_docs(terms, terms_output_file)
    with open(lexicon_output_file, "w") as f:
        json.dump(lexicon, f, indent=4)
    
    print("Processing Life Biology Sentences")
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
            
        # only add chapters 1-10 to subset used for kb matching
        spacy_sent = nlp(re.sub("^([\d*|summary]\.*)+\s*", "", sent))
        if int(sent.split(".")[1]) <= 10:
            sentences_kb_spacy.append(spacy_sent)
        sentences_spacy.append(spacy_sent)
        
    write_spacy_docs(sentences_spacy, life_output_file)
    write_spacy_docs(sentences_kb_spacy, life_kb_output_file)
    