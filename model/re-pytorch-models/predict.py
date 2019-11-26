import argparse
import torch
import numpy as np
import json
import warnings
import os
import sys
import stanfordnlp
from tqdm import tqdm
from utils import tag_relations
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import get_word_pair_classifications, compute_relation_metrics
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage


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
        text = f.read()
    with open(terms, "r") as f:
        terms = f.readlines()
        
    # build input term pair bags
    terms = [nlp(term) for term in terms]
    doc = nlp(text)
    bags = {"no-relation": []}
    for s in doc.sents:
        bags = tag_relations(s, terms, bags, nlp)
    
    # write out to tmp file for loading which we delete later
    tmp_input_file = "./relations_tmp.json"
    with open(tmp_input_file, "w") as f:
        json.dump(bags, f)
    
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

            output = model(batch_data, evaluate=True)
            for i in range(batch_data["data"].shape[0]):
                predictions[batch_data["word_pair"][i]] = {}
                ix = np.argsort(np.array(output[i, :]))[::-1]
                predictions[batch_data["word_pair"][i]]["relations"] = [relations[j] for j in ix] 
                predictions[batch_data["word_pair"][i]]["confidence"] = [10**output[i, j].item() for j in ix] 
                
    filename = f"{out_dir}/{model_version}-relation-predictions.json"
    with open(filename, "w") as f:
        json.dump(predictions, f, indent=4)
        
    os.remove(tmp_input_file)


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

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    model_version = "-".join(args.resume.split("/")[-3:])[:-4]
    
    main(config, args.input_text, args.terms, args.output_dir, model_version)
