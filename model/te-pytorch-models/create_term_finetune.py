import argparse
import torch
import json
from utils import tag_terms, postprocess_tagged_terms, merge_term_results
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.metric import get_term_predictions, compute_term_categories 
from parse_config import ConfigParser
from collections import Counter
import sys
import warnings
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import os

"""
Creates new training data based on human validation input
Expects a file which contains terms that are false positives.
Each line in the file should be a false positive term
"""

def extract_false_term_sentences(input_file, false_terms_set, nlp):
    valid_POS = ["ADJ", "NOUN", "PROPN", "VERB"]
    false_terms = []
 
    for term in false_terms_set:
        spacy_term = nlp(term, disable=["ner", "parser"])
        if all([t.pos_ in valid_POS for t in spacy_term]):
            false_terms.append(spacy_term)
    
    print("Num false terms:", len(false_terms))
    with open(input_file, "r") as f:
        lines = f.readlines()
    spacy_text = []
    result = {}

   
    print("Run spacy tokenizer on text")
    for line in tqdm(lines):
        if line == "\n":
            spacy_text.append("\n")
            continue

        doc = nlp(line, disable=["ner", "parser"])
        spacy_text.append(doc)
    
    filtered_lines = []
    offset = 0
    index = 0
    print("Filtering")
    for text in tqdm(spacy_text):
        if text == "\n":
            filtered_lines.append(lines[index])
            index += 1
            continue
            
        tmp_false = tag_terms(text, false_terms, nlp)
        if len(tmp_false) == 0:
            index += 1
            continue
      
        num_tokens = len(tmp_false["tokenized_text"])
        result = merge_term_results(result, tmp_false["found_terms"], offset)
        offset += num_tokens
        
        filtered_lines.append(lines[index])
        index += 1

    for term in result:
        term_text_set = set(result[term]["text"])
        false_terms_set.update(term_text_set)
    return filtered_lines

def term_model_predict(config, logger):
    """ Uses the specified TermNER model in config to make term predictions for input text
        specified in the main script.
    """

    # setup data_loader instance to load in our tmp file
    data_loader = config.init_obj('data_loader', module_data, split="tmp", data_dir=".", 
                                  shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    # load pre-trained model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    epoch_target = []
    epoch_pred = []
    epoch_terms = Counter() 

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            batch_data["data"] = batch_data["data"].to(device)
            batch_data["target"] = batch_data["target"].to(device)
            batch_data["pad_mask"] = batch_data["pad_mask"].to(device)
            batch_data["bert_mask"] = batch_data["bert_mask"].to(device)

            if len(batch_data["target"].shape) < 2:
                batch_data["target"] = batch_data["target"].unsqueeze(0)

            output = model(batch_data)
            if config["arch"]["type"] == "BertCRFNER": 
                pred = model.decode(output, batch_data["bert_mask"])
            else:
                pred = torch.argmax(output, dim=-1)
            term_predictions = get_term_predictions(pred, batch_data["target"], 
                                                    batch_data["bert_mask"], 
                                                    batch_data["sentences"], data_loader.tags)
            epoch_target += term_predictions["target"]
            epoch_pred += term_predictions["prediction"]
            epoch_terms.update(term_predictions["predicted_terms"])

    return epoch_terms

def main(config, input_file, out_dir, false_term_file, model_version):
    if not false_term_file:
        print("Term file not provided")
        return 
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    warnings.filterwarnings('ignore')
    sys.stdout = open(os.devnull, "w")
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    sys.stdout = sys.__stdout__
    
        
    print("Filtered out sentences only false terms appear")

    false_terms_set = set()
    with open(false_term_file) as f:
        false_terms = f.readlines()
        false_terms = [term.strip() for term in false_terms]
        false_terms_set = set([term for term in false_terms if term != ''])

    # Extracts sentences with only false terms. Also update false_terms set to include non-lemmatized forms 
    lines = extract_false_term_sentences(input_file, false_terms_set, nlp)

    print("Preprocessing Input Text")
    spacy_text = []
    input_data = {"sentences": [], "tags": [], "textbook": [], "terms": {}}
    for line in tqdm(lines):
        if line == "\n":
            spacy_text.append("\n")
            continue
        doc = nlp(line, disable=["ner", "parser"])
        spacy_text.append(doc)
    
        for s in doc.sents:
            tokens = [t.text for t in s]
            input_data["sentences"].append(" ".join(tokens))

            # add in fake data for other entries
            input_data["tags"].append(" ".join(["O"] * len(tokens)))
            input_data["textbook"].append("n/a")

    tmp_input_file = "./term_extraction_tmp.json"
    with open(tmp_input_file, "w") as f:
        json.dump(input_data, f)
    predicted_terms = term_model_predict(config, logger)
    os.remove(tmp_input_file)
        
    predicted_terms = list(predicted_terms.keys())
    predicted_terms = [term for term in predicted_terms if term not in false_terms_set]
    predicted_terms = [nlp(term, disable=["ner", "parser"]) for term in predicted_terms]
    
    # filter out invalid POS
    valid_POS = ["ADJ", "NOUN", "PROPN", "VERB"]
    predicted_terms = [term for term in predicted_terms if all([t.pos_ in valid_POS for t in term])]
    
    term_extraction_data = {"terms" : {}, "sentences": [], "tags" : [], "textbook": []}
    textbook = "n/a"
    print("Retrieving term data for finetuning")
    for text in tqdm(spacy_text):
        if text == "\n":
            continue
            
        tmp = tag_terms(text, predicted_terms, nlp)
        num_tokens = len(tmp["tokenized_text"])
        tmp_terms = list(tmp["found_terms"].keys())
        if len(tmp_terms) == 0:
            continue
        tokenized_sentence = tmp["tokenized_text"]
        term_info = tmp["found_terms"]
        tagged_sentence = tmp["tags"]
        for term in term_info:
            if term in term_extraction_data["terms"]:
                term_extraction_data["terms"][term] += len(term_info[term]["indices"])
            else:
                term_extraction_data["terms"][term] = len(term_info[term]["indices"])
        term_extraction_data["sentences"].append(" ".join(tokenized_sentence))
        term_extraction_data["tags"].append(" ".join(tagged_sentence))
        term_extraction_data["textbook"].append(textbook)

    
    input_filename = input_file.split("/")[-1][:-4]
    # write out predicted terms and annotated text
    filename = f"{out_dir}/{input_filename}_{model_version}_new_data.json"
    with open(filename, "w") as f:
        json.dump(term_extraction_data, f, indent=4)
     

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output_dir', default=".", type=str,
                      help='output directory to place predictions')
    args.add_argument('-i', '--input_file', default=None, type=str,
                      help='input file containing text to make predictions on')
    args.add_argument('-t', '--term_file', default=None, type=str,
                      help='term file containing false positive terms to use')
    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()
    model_version = "-".join(args.resume.split("/")[-3:-1])
    
    main(config, args.input_file, args.output_dir, args.term_file, model_version)
