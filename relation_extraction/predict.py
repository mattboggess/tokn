import argparse
import torch
import json
import torch.nn.functional as F
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from label.label_constants import label_classes
from parse_config import ConfigParser
import numpy as np
from collections import Counter
import spacy
import pandas as pd
import os
import sys
import re
from nltk import sent_tokenize
sys.path.append("../preprocessing")
from data_processing_utils import tag_terms, get_closest_match

invalid_pos = ['JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP']
invalid_dep = ['npadvmod', 'compound', 'amod', 'nmod']

def relation_model_predict(config, logger):
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
    model = config.init_obj('arch', module_arch, 
                            num_classes=len(label_classes),
                            vocab_size=len(data_loader.dataset.tokenizer))

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
        
        epoch_loss = []
        epoch_pred = []
        epoch_label = []
        epoch_score = []
        relation_classes = label_classes 
        
        for i, batch_data in enumerate(tqdm(data_loader)):
            
            for field in ['input_ids', 'label', 'target', 'attention_mask', 'term1_mask', 'term2_mask']:
                batch_data[field] = batch_data[field].to(device)
            
            output = model(batch_data)
            pred = torch.argmax(output, dim=-1)
            
            # accumulate epoch quantities 
            epoch_score += list(F.softmax(output, dim=-1).cpu().detach().numpy().max(axis=-1))
            epoch_pred += [relation_classes[p.item()] for p in pred]
        
     # save out predictions for every instance 
    data = data_loader.dataset.data
    data['predicted_relation'] = epoch_pred
    data['prediction_confidence'] = [np.max(s) for s in epoch_score]
    
    return data
                
def preprocess_input_text(data, terms, nlp):
    """
    Preprocesses input text so that it is a pandas dataframe in the expected format for the model.
    """
    new_df = []
    for k, row in tqdm(list(data.iterrows())):
        tmp = tag_terms(row.sentence, terms, nlp, invalid_dep=invalid_dep, invalid_pos=invalid_pos)
        found_terms_info = tmp['found_terms']
        found_terms = list(found_terms_info.keys())
        found_term_pairs = [] 
        for i in range(len(found_terms) - 1):
            for j in range(i + 1, len(found_terms)):
                term_pair = (found_terms[i], found_terms[j])

                indices = get_closest_match(
                    found_terms_info[term_pair[0]]['indices'],
                    found_terms_info[term_pair[1]]['indices']
                )

                new_row = row.copy()
                if indices[0][0] > indices[1][0]:
                    term_pair = (term_pair[1], term_pair[0])
                    indices = (indices[1], indices[0])
                new_row['term_pair'] = term_pair
                new_row['term1'] = term_pair[0]
                new_row['term2'] = term_pair[1]
                new_row['term1_location'] = indices[0]
                new_row['term2_location'] = indices[1]
                new_row['tokens'] = tmp['tokenized_text']
                new_df.append(new_row)
                
    data = pd.DataFrame(new_df)
    
    # fake labels
    data['hard_label'] = 1
    data['hard_label_class'] = (label_classes * data.shape[0])[:data.shape[0]]
    data['soft_label'] = 1
    return data

def main(config, input_file, output_dir, term_file, model_version):
    logger = config.get_logger('test')
    
    # set up spacy nlp engine
    nlp = spacy.load('en_core_web_sm')
    
    # read in sentences 
    if input_file.endswith('.csv'):
        input_type = 'csv'
        data = pd.read_csv(input_file)
    else: 
        input_type = 'text'
        with open(input_file, "r") as f:
            text = f.read()
            sentences = sent_tokenize(text)
            data = pd.DataFrame({'sentence': sentences})
          
    # read in terms 
    print("Processing Terms")
    if term_file.endswith('.csv'):
        terms = pd.read_csv(term_file).term
    else:
        with open(term_file, 'r') as fid:
            terms = fid.readlines()
    terms = [t.strip() for t in terms if t.strip() != '']
    terms = [nlp(t) for t in terms]
        
    print("Preprocessing Input Text")
    input_data = preprocess_input_text(data, terms, nlp)
    
    # write out to tmp file compatible with model for loading
    tmp_input_file = "./tmp.pkl"
    input_data.to_pickle(tmp_input_file)
    
    # predict terms using saved model
    print("Predicting terms using model")
    predicted_relations = relation_model_predict(config, logger)
    os.remove(tmp_input_file)
    
    # write out sentence predictions
    input_filename = input_file.split('/')[-1][:-4]
    predicted_relations = predicted_relations[['term_pair', 'predicted_relation', 'prediction_confidence', 'sentence']]
    predicted_relations.to_csv(f"{output_dir}/{input_filename}_{model_version}_sentence_predictions.csv", index=False)
    
    # write out term pair predictions
    tp_preds = {}
    rel_preds = predicted_relations \
         .groupby(['predicted_relation', 'term_pair']) \
         .prediction_confidence.mean() \
         .reset_index()
    rel_preds = rel_preds[rel_preds.predicted_relation != 'OTHER']
    rel_preds = rel_preds.sort_values(['predicted_relation', 'prediction_confidence'], ascending=False)
    rel_preds.to_csv(f"{output_dir}/{input_filename}_{model_version}_term_pair_predictions.csv", index=False)
            
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
