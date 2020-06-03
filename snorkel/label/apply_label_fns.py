# Applies snorkel label functions to a train, dev, and test tagged term pair sentences split.
#
# Author: Matthew Boggess
# Version: 4/27/20

# Data Source: 
#   - Output train, dev, and test splits from preprocessing/split_data.py

# Description: 
#   - Runs each split through the following labelling pipeline:
#     - Spacy preprocesses each split to be used by the label functions
#     - Applies each labelling function to each row of the dataframe using the snorkel applier.
#       Saves out the accompanying analysis
#     - Computes majority vote labels for each split using the labelling function
#     - Computes probabilistic labels for each split using the Snorkel LabelModel fit to the train
#     - Writes out modified pandas dataframes for each split with these labels added

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

# assemble label functions
#relation_class = "meronym" # meronym, taxonomy, or all
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
        data = data.drop_duplicates(['sentence', 'term1', 'term2'])
        
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
        
        # add label fn and kb specific aggregate labels
        
        labelled_data[split] = data
    
    label_model = LabelModel(cardinality=class_card, verbose=True)
    label_model.fit(L_train=labels['train'], n_epochs=10, log_freq=100, seed=123)
    
    for split in splits:
        print(f"Label model labelling {split} data")
        lm_preds = label_model.predict_proba(labels[split])
        labelled_data[split]['label_model_labels'] = [x.tolist() for x in lm_preds]
        
        # write out fully labelled version of data
        with open(f"{label_output_data_dir}/{split}_labelled.pkl", 'wb') as fid:
            pickle.dump(labelled_data[split].drop('doc', axis=1), fid)
            
        print(f"Prepping {split} data for modelling")
        # add in gold standard labels
        if split in ['dev', 'test']:
            gold_df = pd.read_csv(f"{input_data_dir}/{split}-manual_eval.csv")
            gold_df = gold_df[['sentence', 'term_pair', 'gold_label']]
            gold_df.term_pair = gold_df.term_pair.apply(literal_eval)
            labelled_data[split] = labelled_data[split].merge(
                gold_df, how='left', on=['sentence', 'term_pair'])
            labelled_data[split]['gold_label'] = labelled_data[split].gold_label.fillna('OTHER')
        
        # standardize hard label columns for model
        if split in ['dev', 'test']:
            labelled_data[split]['hard_label'] = labelled_data[split].gold_label.apply(
                lambda x: label_classes.index(x))
        else: 
            labelled_data[split]['hard_label'] = labelled_data[split]['majority_vote']
        labelled_data[split]['label_class'] = labelled_data[split].hard_label.apply(
            lambda x: add_label(x))
        
        # stanardize soft label columns for model
        labelled_data[split]['soft_label'] = labelled_data[split].label_model_labels
        
        # filter out abstained training data
        labelled_data[split] = labelled_data[split][labelled_data[split].hard_label >= 0]
        
        # write out model version of data
        labelled_data[split].drop('doc', axis=1).to_pickle(f"{model_output_dir}/{split}.pkl")
        
        # create a debug set for model
        if split == 'train':
            train = labelled_data[split]
            sent_check = train.groupby('sentence').agg(
                {'hard_label': lambda x: sum(x) > 0}).reset_index()
            pos_exs = sent_check[sent_check.hard_label].sample(10)
            neg_exs = sent_check[~sent_check.hard_label].sample(10)
            debug = pd.concat(
                [train[train.sentence.isin(pos_exs.sentence)], 
                 train[train.sentence.isin(neg_exs.sentence)]], 
                axis=0)
            debug.drop('doc', axis=1).to_pickle(f"{model_output_dir}/debug.pkl")
            


        
        
            
            
        
    
    
        
