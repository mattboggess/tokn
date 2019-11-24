import argparse
import torch
import json
from utils import tag_terms
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


def main(config, input_file, out_dir, model_version):
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    warnings.filterwarnings('ignore')
    sys.stdout = open(os.devnull, "w")
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    sys.stdout = sys.__stdout__
    
    # read in text data
    with open(input_file, "r") as f:
        text = f.read()
    
    logger.info("Preprocessing Input Text")
    input_data = {"sentences": [], "tags": [], "textbook": [], "terms": {}}
    doc = nlp(text)
    for s in doc.sents:
        tokens = [t.text for t in s]
        input_data["sentences"].append(" ".join(tokens))
        
        # add in fake data for other entries
        input_data["tags"].append(" ".join(["O"] * len(tokens)))
        input_data["textbook"].append("n/a")
    
    # write out to tmp file for loading which we delete later
    tmp_input_file = "./term_extraction_tmp.json"
    with open(tmp_input_file, "w") as f:
        json.dump(input_data, f)

    # setup data_loader instance to load in our tmp file
    data_loader = config.init_obj('data_loader', module_data, split="tmp", data_dir=".", shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    model = model.to(device)
    model.eval()

    epoch_target = []
    epoch_pred = []
    epoch_terms = Counter() 
    
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
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

    
    predicted_terms = list(epoch_terms.keys())
    result = tag_terms(text, predicted_terms, nlp)
    
    filename = f"{out_dir}/{model_version}_predicted_terms.json"
    with open(filename, "w") as f:
        json.dump(result["found_terms"], f)
        
    filename = f"{out_dir}/{model_version}_annotated_text.txt"
    with open(filename, "w") as f:
        f.write(result["annotated_text"])
    
    os.remove(tmp_input_file)
    logger.info(result["found_terms"])


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

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    model_version = "-".join(args.resume.split("/")[-3:])[:-4]
    
    main(config, args.input_file, args.output_dir, model_version)
