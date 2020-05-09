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

#===================================================================================
# Parameters

input_data_dir = "../data"
output_data_dir = "../data/label"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

label_fns = [
    isa_pattern, 
    suchas_pattern, 
    including_pattern, 
    called_pattern, 
    especially_pattern,
    appo_pattern, 
    other_pattern, 
    are_pattern,
    whichis_pattern,
    term_part_of_speech,
    term_dep_role,
    term_modifier,
    term_subset,
    kb_bio101_ds_taxonomy,
    kb_bio101_ds_synonym,
    kb_bio101_ds_negative,
    nsubj_pattern
]

# fix to extract name only in str form for each label function
label_fn_names = [re.match('.*Function (.+), .*', str(lf)).group(1) for lf in  label_fns]

# number of different classes
class_card = len(label_classes)

splits = ['test', 'dev', 'train']

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    applier = PandasLFApplier(lfs=label_fns)
    labelled_data = {}
    labels = {}
    for split in splits:
        
        with open(f"{input_data_dir}/{split}.pkl", 'rb') as fid:
            data = pickle.load(fid)
        
        print(f"Spacy preprocessing {split} data")
        # only spacy preprocess each unique sentence once then merge into full dataframe
        docs = []
        sent_data = data.groupby('text').chapter.count().reset_index().drop('chapter', axis=1)
        for _, row in tqdm(list(sent_data.iterrows())):
            docs.append(nlp(row.text))
        sent_data['doc'] = docs
        data = data.merge(sent_data, on=['text'])
        
        print(f"Applying label functions to {split} data")
        labels[split] = applier.apply(df=data)
        label_df = pd.DataFrame(labels[split], columns=label_fn_names)
        data = pd.concat([data, label_df], axis=1)
        lf_analysis = LFAnalysis(L=labels[split], lfs=label_fns).lf_summary()
        lf_analysis.to_csv(f"{output_data_dir}/{split}_label_analysis.csv")
        
        print(f"Majority vote labelling {split} data")
        majority_model = MajorityLabelVoter(cardinality=class_card)
        majority_labels = majority_model.predict(L=labels[split])
        data['majority_vote'] = majority_labels
        
        labelled_data[split] = data
    
    print("Fitting Label Model to Training Data")
    label_model = LabelModel(cardinality=class_card, verbose=True)
    label_model.fit(L_train=labels['train'], n_epochs=500, log_freq=100, seed=123)
    
    for split in splits:
        print(f"Label model labelling {split} data")
        lm_preds = label_model.predict_proba(labels[split])
        labelled_data[split]['label_model_labels'] = [x.tolist() for x in lm_preds]
        
        with open(f"{output_data_dir}/{split}_labelled.pkl", 'wb') as fid:
            pickle.dump(labelled_data[split].drop('doc', axis=1), fid)
            
            
        
    
    
        
