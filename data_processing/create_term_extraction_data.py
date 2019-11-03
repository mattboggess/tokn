from data_processing_utils import tag_terms, read_spacy_docs
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import warnings
import os
import json
import re


if __name__ == "__main__":
    
    # initialize Stanford NLP Spacy pipeline
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    warnings.filterwarnings('ignore')
    
    input_data_dir = "../data/preprocessed_data"
    output_data_dir = "../data/term_extraction"
    
    splits = ["train", "validation", "life_test", "psych_test"]
    
    # load in key terms
    files = os.listdir(input_data_dir)
    term_files = [f for f in files if "key_terms_spacy" in f]
    terms = {split: [] for split in splits}
    full_terms = []
    for file in term_files:
        input_terms = read_spacy_docs(f"{input_data_dir}/{file}", nlp)
        if "Life_Biology" not in file:
            full_terms += input_terms
        lemmas = [" ".join(t.lemma_ for t in term) for term in input_terms]
        if "Life_Biology" in file:
            terms["life_test"] += lemmas 
        elif "Psychology" in file:
            terms["psych_test"] += lemmas
        elif "Biology_2e" in file:
            terms["validation"] += lemmas 
        else:
            terms["train"] += lemmas 
    
    # de-dupe key terms
    seen_lemmas = set() 
    deduped_terms = []
    for term in full_terms:
        lemma = " ".join([t.lemma_ for t in term])
        if lemma not in seen_lemmas:
            deduped_terms.append(term)
            seen_lemmas.add(lemma)
    print(f"Found {len(deduped_terms)} unique terms")
    
    # load in and tag sentences
    term_extraction_data = {
        "train": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "validation": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "life_test": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "psych_test": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
    }
    sentence_files = [f for f in files if "sentences_spacy" in f]
    for file in sentence_files:
        textbook = re.match("(.*)_sentences_spacy", file).group(1)
        print(f"Tagging {textbook}")
        if "Life_Biology" in file:
            split = "life_test"
        elif "Psychology" in file:
            split = "psych_test"
        elif "Biology_2e" in file:
            split = "validation"
        else:
            split = "train"
        
        sentences = read_spacy_docs(f"{input_data_dir}/{file}", nlp)
        for sentence in sentences:
            tokenized_sentence, tagged_sentence, term_info = tag_terms(sentence, deduped_terms, nlp)
            if len(term_info.keys()) == 0:
                continue
            
            # prevent overlap in splits
            if split == "train":
                overlap = [term in terms["psych_test"] for term in term_info] + \
                          [term in terms["validation"] for term in term_info]
            else:
                overlap = [term in terms["train"] for term in term_info]
            if any(overlap):
                continue
                
            for term in term_info:
                if term in term_extraction_data[split]["terms"]:
                    term_extraction_data[split]["terms"][term] += len(term_info[term]["indices"])
                else:
                    term_extraction_data[split]["terms"][term] = len(term_info[term]["indices"])
            term_extraction_data[split]["sentences"].append(" ".join(tokenized_sentence))
            term_extraction_data[split]["tags"].append(" ".join(tagged_sentence))
            term_extraction_data[split]["textbook"].append(textbook)
            
    # write out data splits
    for split in term_extraction_data:
        with open(f"{output_data_dir}/term_extraction_{split}.json", "w") as f:
            json.dump(term_extraction_data[split], f)
        
        