import argparse
import torch
import numpy as np
import json
import warnings
import os
import sys
import stanfordnlp
from tqdm import tqdm
from utils import tag_relations, postprocess_relation_predictions
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import get_word_pair_classifications, compute_relation_metrics
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

def relation_model_predict(config, logger):
    
    # setup data_loader instance to load in our tmp file
    data_loader = config.init_obj('data_loader', module_data, split="tmp", data_dir=".", 
                                  shuffle=False, predict=True)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    # load trained model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    
    # prepare model for testing
    device = "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = {}
    relations = data_loader.dataset.relations
    
    with torch.no_grad():
        
        for i, batch_data in enumerate(tqdm(data_loader)):
            for field in ["data", "pad_mask", "e1_mask", "e2_mask", "sentence_mask"]:
                batch_data[field] = batch_data[field].to(device)

            output, prob = model(batch_data, evaluate=True)
            for i in range(batch_data["data"].shape[0]):
                predictions[batch_data["word_pair"][i]] = {}
                ix = np.argsort(np.array(output[i, :]))[::-1]
                predictions[batch_data["word_pair"][i]]["relations"] = [relations[j] for j in ix] 
                predictions[batch_data["word_pair"][i]]["confidence"] = [prob[i, j].item() for j in ix] 
    
    return predictions

def main(config, input_text, terms, out_dir, model_version):
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    warnings.filterwarnings('ignore')
    sys.stdout = open(os.devnull, "w")
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    sys.stdout = sys.__stdout__
    
    # read in text and terms 
    with open(input_text, "r") as f:
        lines = f.readlines()
    if terms.endswith(".txt"):
        with open(terms, "r") as f:
            terms = f.readlines()
    elif terms.endswith(".json"):
        with open(terms, "r") as f:
            terms = list(json.load(f).keys())
        
        
    # build input term pair bags
    terms = [nlp(term, disable=["ner", "parser"]) for term in terms]
    bags = {"no-relation": []}
    print("Preprocessing Data")
    for line in tqdm(lines):
        if len(line.strip()) == 0:
            continue
        doc = nlp(line, disable=["ner", "parser"])
        for sent in doc.sents:
            bags = tag_relations(sent, terms, bags, nlp)
    
    # write out to tmp file for loading which we delete later
    tmp_input_file = "./relations_tmp.json"
    with open(tmp_input_file, "w") as f:
        json.dump(bags, f)
    
    print("Predicting Relations")
    predictions = relation_model_predict(config, logger)
    predictions = postprocess_relation_predictions(predictions)
    
    os.remove(tmp_input_file)
                
    input_filename = input_text.split("/")[-1][:-4]
    filename = f"{out_dir}/{input_filename}_{model_version}_predicted_relations.json"
    with open(filename, "w") as f:
        json.dump(predictions, f, indent=4)
        


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
    args.add_argument('-i', '--input_text', default=None, type=str,
                      help='input file containing text to make predictions on')
    args.add_argument('-t', '--terms', default=None, type=str,
                      help='input file containing terms that will be used')

    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()
    model_version = "-".join(args.resume.split("/")[-3:-1])
    
    main(config, args.input_text, args.terms, args.output_dir, model_version)
