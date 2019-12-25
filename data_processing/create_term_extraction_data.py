##########
# Processes openstax and life biology textbooks by tagging all key terms from glossaries/KB
# in each sentence. Splits data into traing and test splits.
##########
from data_processing_utils import tag_terms, read_spacy_docs
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import warnings
import os
import json
import re
import copy


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
        if "Life_Biology_kb" not in file:
            full_terms += input_terms
        else:
            life_bio_terms = input_terms
        lemmas = [" ".join(t.lemma_ for t in term) for term in input_terms]
        
        if "Life_Biology_kb" in file:
            terms["life_test"] += lemmas 
        elif "Psychology" in file:
            terms["psych_test"] += lemmas
        elif "Biology_2e" in file:
            terms["validation"] += lemmas 
        else:
            terms["train"] += lemmas 
    
    print(f"Found {len(full_terms)} openstax terms")
    print(f"Found {len(life_bio_terms)} life biology terms")
    
    # load in and tag sentences
    term_extraction_data = {
        "train": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "validation": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "life_test": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "psych_test": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "full": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
        "debug": {"terms": {}, "sentences": [], "tags": [], "textbook": []},
    }
    sentence_files = [f for f in files if "sentences_spacy" in f]
    for file in sentence_files:
        
        # skip the full life bio sentences to focus on the KB
        if file == "Life_Biology_sentences_spacy":
            continue
            
        textbook = re.match("(.*)_sentences_spacy", file).group(1)
        print(f"Tagging {textbook}")
        if "Life_Biology_kb" in file:
            split = "life_test"
            check_terms = life_bio_terms
        elif "Psychology" in file:
            split = "psych_test"
            check_terms = full_terms 
        elif "Biology_2e" in file:
            split = "validation"
            check_terms = full_terms 
        else:
            split = "train"
            check_terms = full_terms 
        
        sentences = read_spacy_docs(f"{input_data_dir}/{file}", nlp)
        for sentence in sentences:
            
            # tag key terms for the given sentence
            result = tag_terms(sentence, check_terms, nlp)
            tokenized_sentence = result["tokenized_text"]
            term_info = result["found_terms"]
            tagged_sentence = result["tags"]
            
            # ignore sentences without any tagged terms
            if len(term_info.keys()) == 0:
                continue
                
            # add sentence to full data split
            if split != "life_test":
                for term in term_info:
                    if term in term_extraction_data["full"]["terms"]:
                        term_extraction_data["full"]["terms"][term] += len(term_info[term]["indices"])
                    else:
                        term_extraction_data["full"]["terms"][term] = len(term_info[term]["indices"])
                term_extraction_data["full"]["sentences"].append(" ".join(tokenized_sentence))
                term_extraction_data["full"]["tags"].append(" ".join(tagged_sentence))
                term_extraction_data["full"]["textbook"].append(textbook)
            
            # create small debug split 
            if len(term_extraction_data["full"]["sentences"]) == 32:
                term_extraction_data["debug"] = copy.deepcopy(term_extraction_data["full"])
            
            # prevent overlap across train/test splits
            if split == "train":
                overlap = [term in terms["psych_test"] for term in term_info] + \
                          [term in terms["validation"] for term in term_info] + \
                          [term in terms["life_test"] for term in term_info]
            else:
                overlap = [term in terms["train"] for term in term_info]
            if any(overlap):
                continue
                
            # add to appropriate train/dev/test split
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
        
        