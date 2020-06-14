# Applies snorkel labelling pipeline to a train, dev, and test set for relation extraction.
#
# Author: Matthew Boggess
# Version: 6/12/20

# Data Source: 
#   - Output train, dev, and test splits from ../preprocessing/split_relation_extraction_data.py

# Description: 
#   - Runs each split through the following labelling pipeline:
#     - Applies each labelling function to each row of the dataframe using the snorkel applier.
#       Saves out the accompanying label fn analysis
#     - Computes majority vote labels for each split using the labelling function
#     - Computes probabilistic labels for each split using the Snorkel LabelModel fit to the train
#     - Writes out modified pandas dataframes for each split with these labels added
#     - Additionally creates modified versions of the split ready for modelling that:
#       - standardizes names for hard and soft labels when predicting
#       - removes training examples that don't have a label
#       - takes a small sample from train to be used for debugging

#===================================================================================
# Libraries

from label_constants import *
from taxonomy_label_fns import *
from meronym_label_fns import *
from kb_bio101_label_fns import *
from other_label_fns import *
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel
import spacy
from tqdm import tqdm
import os
import pickle
import pandas as pd
import re
from ast import literal_eval

#===================================================================================
# Parameters

input_data_dir = "../../data/relation_extraction"
label_output_data_dir = "../../data/relation_extraction/label"
model_output_data_dir = "../../data/relation_extraction/model"

# list of all label functions we will apply to the data 
label_fns = [kb_bio101_ds_negative, term_part_of_speech] + meronym_label_fns + meronym_kb_fns + \
            taxonomy_kb_fns + taxonomy_label_fns
# fix to extract name only in str form for each label function
label_fn_names = [re.match('.*Function (.+), .*', str(lf)).group(1) for lf in  label_fns]
class_card = len(label_classes)

splits = ['test', 'dev', 'train']

#===================================================================================
# Helper Functions

def add_label(x):
    if x == -1:
        return 'ABSTAIN'
    else:
        return label_classes[x]

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    applier = PandasLFApplier(lfs=label_fns)
    labelled_data = {}
    labels = {}
    for split in splits:
        
        data = pd.read_pickle(f"{input_data_dir}/{split}.pkl").reset_index()
        data = data.drop_duplicates(['sentence', 'term1', 'term2']).reset_index()
        
        print(f"Applying label functions to {split} data")
        labels[split] = applier.apply(df=data)
        label_df = pd.DataFrame(labels[split], columns=label_fn_names)
        data = pd.concat([data, label_df], axis=1)
        lf_analysis = LFAnalysis(L=labels[split], lfs=label_fns).lf_summary()
        lf_analysis.to_csv(f"{label_output_data_dir}/{split}_label_analysis.csv")
        
        print(f"Majority vote labelling {split} data")
        majority_model = MajorityLabelVoter(cardinality=class_card)
        majority_labels = majority_model.predict(L=labels[split])
        data['majority_vote'] = majority_labels
        data['majority_vote_relation'] = data.majority_vote.apply(lambda x: human_label(x, label_classes))
        
        # add label fn and kb specific aggregate label columns
        label_fn_cols = [col for col in data.columns if col.endswith('lf')]
        kb_label_cols = [col for col in data.columns if col.startswith('kb')]
        data['label_fn_label'] = data.apply(lambda row: max([row[c] for c in label_fn_cols]), axis=1)
        data['kb_label'] = data.apply(lambda row: max([row[c] for c in kb_label_cols]), axis=1)
        data.label_fn_label = data.label_fn_label.apply(lambda x: human_label(x, label_classes))
        data.kb_label = data.kb_label.apply(lambda x: human_label(x, label_classes))
        
        labelled_data[split] = data
    
    label_model = LabelModel(cardinality=class_card, verbose=True)
    label_model.fit(L_train=labels['train'], n_epochs=10, log_freq=100, seed=123)
    
    for split in splits:
        print(f"Label model labelling {split} data")
        lm_preds = label_model.predict_proba(labels[split])
        labelled_data[split]['label_model_labels'] = [x.tolist() for x in lm_preds]
        
        # write out fully labelled versions of data
        with open(f"{label_output_data_dir}/{split}_labelled.pkl", 'wb') as fid:
            pickle.dump(labelled_data[split].drop('doc', axis=1), fid)
            
        print(f"Prepping {split} data for modelling")
        
        # standardize hard label columns for model
        if split in ['dev', 'test']:
            labelled_data[split]['hard_label'] = labelled_data[split].gold_label.apply(
                lambda x: label_classes.index(x))
        else: 
            labelled_data[split]['hard_label'] = labelled_data[split]['majority_vote']
        labelled_data[split]['hard_label_class'] = labelled_data[split].hard_label.apply(
            lambda x: add_label(x))
        
        # stanardize soft label columns for model
        labelled_data[split]['soft_label'] = labelled_data[split].label_model_labels
        
        # filter out abstained training data
        if split == 'train':
            train_size = labelled_data[split].shape[0]
            labelled_data[split] = labelled_data[split][labelled_data[split].hard_label >= 0]
            reduced_train_size = labelled_data[split].shape[0]
            print(f"Labels cover {reduced_train_size / train_size} of training data")
        
        # write out model ready versions of data
        labelled_data[split].drop('doc', axis=1).to_pickle(f"{model_output_data_dir}/{split}.pkl")
        
        # create a debug set for model
        if split == 'train':
            train = labelled_data[split]
            debug = train.groupby('hard_label').apply(lambda x: x.sample(10)).reset_index(drop=True)
            debug.drop('doc', axis=1).to_pickle(f"{model_output_data_dir}/debug.pkl")
            


        
        
            
            
        
    
    
        
