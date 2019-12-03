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

def main(config, input_file, out_dir, term_file, model_version):
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    warnings.filterwarnings('ignore')
    sys.stdout = open(os.devnull, "w")
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    sys.stdout = sys.__stdout__
    
    # read in text data
    with open(input_file, "r") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        
    print("Preprocessing Input Text")
    input_data = {"sentences": [], "tags": [], "textbook": [], "terms": {}}
    spacy_text = []
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
            
        
    # load pre-specified terms if provided
    if term_file:
        print("Using Pre-Specified Terms")
        with open(term_file, "r") as f:
            predicted_terms = json.load(f)
    # get terms via model predictions
    else:
        print("Predicting terms using model")
        # write out to tmp file compatible with model for loading
        tmp_input_file = "./term_extraction_tmp.json"
        with open(tmp_input_file, "w") as f:
            json.dump(input_data, f)
            
        predicted_terms = term_model_predict(config, logger)
        
        os.remove(tmp_input_file)
            
    # tag/annotate the text with our predicted terms
    print("Annotating Input Text")
    predicted_terms = list(predicted_terms.keys())
    predicted_terms = [nlp(term, disable=["ner", "parser"]) for term in predicted_terms]
    
    # filter out invalid POS
    valid_POS = ["ADJ", "NOUN", "PROPN", "VERB"]
    predicted_terms = [term for term in predicted_terms if all([t.pos_ in valid_POS for t in term])]
    
    result = {"found_terms": {}, "annotated_text": ""}
    offset = 0
    for text in tqdm(spacy_text):
        if text == "\n":
            result["annotated_text"] += "\n" 
            continue
            
        tmp = tag_terms(text, predicted_terms, nlp)
        num_tokens = len(tmp["tokenized_text"])
        tmp = postprocess_tagged_terms(tmp)
        
        result["annotated_text"] = "\n".join([result["annotated_text"], tmp["annotated_text"]])
        result["found_terms"] = merge_term_results(result["found_terms"], tmp["found_terms"], offset)
        offset += num_tokens 
    
    # write out predicted terms and annotated text
    input_filename = input_file.split("/")[-1][:-4]
    filename = f"{out_dir}/{input_filename}_{model_version}_predicted_terms.json"
    with open(filename, "w") as f:
        json.dump(result["found_terms"], f, indent=4)
        
    filename = f"{out_dir}/{input_filename}_{model_version}_annotated_text.txt"
    with open(filename, "w") as f:
        f.write(result["annotated_text"].strip("\n"))
    

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
                      help='optional term file containing terms to use instead of model predictions')

    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()
    model_version = "-".join(args.resume.split("/")[-3:-1])
    
    main(config, args.input_file, args.output_dir, args.term_file, model_version)
