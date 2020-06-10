import argparse
import torch
import json
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.metric import get_term_predictions, compute_term_categories 
from parse_config import ConfigParser
import numpy as np
from collections import Counter
import spacy
import pandas as pd
import os
import sys
from nltk import sent_tokenize
sys.path.append("../preprocessing")
from data_processing_utils import tag_terms

invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP']
invalid_dep = ['npadvmod', 'compound', 'amod', 'nmod']

def determine_term_type(term):
    """ Categorizes a term as either entity or event based on several derived rules.

    Parameters
    ----------
    term: spacy.tokens.doc.Doc
        Spacy preprocessed representation of the term 

    Returns
    -------
    str ('entity' | 'event')
        The class of the term
    """
    
    NOMINALS = ["ation", "ition", "ption", "ing", "sis", "lism", "ment", "sion"]
    EVENT_KEYWORDS = ["process", "cycle"]
    
    # key words that indicate events despite being nouns 
    if any([ek in term.text.lower() for ek in EVENT_KEYWORDS]):
        term_type = "event"
    # key endings indicating a nominalized form of an event 
    elif any([term[i].text.endswith(ne) for ne in NOMINALS for i in range(len(term))]):
        term_type = "event"
    # POS = Verb implies event 
    elif any([t.pos_ == "VERB" for t in term]):
        term_type = "event"
    # default is otherwise entity 
    else:
        term_type = "entity"
    
    return term_type

def merge_term_results(results1, results2, offset=0):
    """ Merges results of term tagging into a single dictionary
        accounting for offset with term indices.
    """
    for term in results2:
        if term not in results1:
            results1[term] = results2[term]
        else:
            for field in ["text", "indices", "pos"]:
                if field == "indices":
                    indices_adj = [(t[0] + offset, t[1] + offset) for t in results2[term][field]]
                    results1[term][field] += indices_adj 
                else:
                    results1[term][field] += results2[term][field]
    return results1

def term_model_predict(config, logger):
    """ Uses the specified TermNER model in config to make term predictions for input text
        specified in the main script.
    """

    # setup data_loader instance to load in our tmp file
    data_loader = config.init_obj(
        'data_loader', 
        module_data, 
        split='tmp', 
        data_dir='.', 
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
    epoch_terms = [] 
    epoch_probs = []

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            batch_data['data'] = batch_data['data'].to(device)
            batch_data['target'] = batch_data['target'].to(device)
            batch_data['pad_mask'] = batch_data['pad_mask'].to(device)
            batch_data['bert_mask'] = batch_data['bert_mask'].to(device)

            if len(batch_data['target'].shape) < 2:
                batch_data['target'] = batch_data['target'].unsqueeze(0)

            output = model(batch_data)
            if config['arch']['type'] == 'BertCRFNER': 
                pred = model.decode(output, batch_data['bert_mask'])
            else:
                pred = torch.argmax(output, dim=-1)
            
             # get predicted terms with probabilities for each sentence
            terms, probs = get_term_predictions(output,  
                                                batch_data['bert_mask'], 
                                                batch_data['sentences'], 
                                                data_loader.tags)
            epoch_terms += terms
            epoch_probs += probs
            
    output_terms = {}
    for terms, probs in zip(epoch_terms, epoch_probs):
        for t, p in zip(terms, probs):
            if t in output_terms:
                output_terms[t].append(p)
            else:
                output_terms[t] = [p]
    
    for t in output_terms:
        output_terms[t] = np.mean(output_terms[t])
            
    return output_terms

def preprocess_input_text(lines, nlp):
    """
    Preprocesses input text so that it is a pandas dataframe in the expected format for the model.
    """
    input_data = {'sentences': [], 'tags': [], 'textbook': [], 'tokens': [], 'doc': []}
    spacy_text = []
    for line in tqdm(lines):
        if line == '\n':
            spacy_text.append('\n')
            continue

        doc = nlp(line)
        spacy_text.append(doc)
        for s in doc.sents:
            tokens = [t.text for t in s]
            input_data['tokens'].append(tokens)
            input_data['sentences'].append(' '.join(tokens))
            input_data['doc'].append(s.as_doc())

            # add in fake data for other entries
            input_data['tags'].append(['O'] * len(tokens))
            input_data['textbook'].append('n/a')
            
    input_data = pd.DataFrame(input_data)
    return input_data, spacy_text

def main(config, input_file, out_dir, term_file, model_version):
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    nlp = spacy.load('en_core_web_sm')
    
    # read in text data
    if input_file.endswith('.csv'):
       data = pd.read_csv(input_file)
       lines = data.sentence
    else: 
        with open(input_file, "r") as f:
            lines = f.readlines()
            text = "\n".join(lines)
        
    print("Preprocessing Input Text")
    input_data, spacy_text = preprocess_input_text(lines, nlp)
    
    # write out to tmp file compatible with model for loading
    tmp_input_file = "./term_extraction_tmp.pkl"
    input_data.to_pickle(tmp_input_file)
    
    # predict terms using saved model
    print("Predicting terms using model")
    predicted_terms = term_model_predict(config, logger)
    os.remove(tmp_input_file)
            
    # tag/annotate the text with our predicted terms
    print("Annotating Input Text")
    tagging_terms = list(predicted_terms.keys())
    tagging_terms = [t for t in tagging_terms if t.strip() != '-']
    tagging_terms = [nlp(t) for t in tagging_terms]    
    result = {"found_terms": {}, "annotated_text": ""}
    offset = 0
    for text in tqdm(spacy_text):
        if text == '\n':
            result['annotated_text'] += '\n' 
            continue
            
        # Tag our predicted terms in the original text 
        tmp = tag_terms(text, tagging_terms, nlp, invalid_dep=invalid_dep, invalid_pos=invalid_pos)
        num_tokens = len(tmp['tokenized_text'])
       
        # Merge results with tmp 
        result['annotated_text'] = '\n'.join([result['annotated_text'], tmp['annotated_text']])
        result['found_terms'] = merge_term_results(result['found_terms'], tmp['found_terms'], offset)
        offset += num_tokens 
    
    # write out annotated text
    input_filename = input_file.split('/')[-1][:-4]
    filename = f"{out_dir}/{input_filename}_{model_version}_annotated_text.txt"
    with open(filename, 'w') as f:
        f.write(result['annotated_text'].strip('\n'))
    
    # write out detailed terms
    filename = f"{out_dir}/{input_filename}_{model_version}_predicted_terms_detailed.json"
    with open(filename, 'w') as f:
        json.dump(result['found_terms'], f, indent=4)
    
    # create simplified representation of terms with confidence scores and entity/event labels
    simplified_terms = {'term': [], 'base_term': [], 'type': [], 'confidence': []}
    for lemma in result['found_terms']:
        for text_rep in set(result['found_terms'][lemma]['text']):
            simplified_terms['term'].append(text_rep)
            simplified_terms['base_term'].append(lemma)
            simplified_terms['type'].append(determine_term_type(nlp(text_rep)))
            simplified_terms['confidence'].append(predicted_terms[lemma])
    simplified_terms = pd.DataFrame(simplified_terms)
    filename = f"{out_dir}/{input_filename}_{model_version}_predicted_terms_simple.csv"
    simplified_terms.sort_values(['base_term']).to_csv(filename, index=False)
        

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
    model_version = '-'.join(args.resume.split('/')[-3:-1])
    
    main(config, args.input_file, args.output_dir, args.term_file, model_version)
