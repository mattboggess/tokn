# Takes in preprocessed sentences and KB relations file 
# Creates separate sentence and key term files for each.
# For Life, does additional parsing to extract out the knowledge base lexicon and first 10 chapters.

# Author: Matthew Boggess
# Version: 4/22/20

# Data Source: 
#   - Outputs from Inquire knowledge base provided by Dr. Chaudhri

# Description: 
#   - Processes a dump from the Inquire knowledge base to produce the following output:
#       A Spacy NLP preprocessed set of biology terms extracted from the first 10 chapters
#       of Life Biology for the previous knowledge base

#===================================================================================
# Libraries

from snorkel_functions.taxonomy_labelers import *
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel
import spacy
from tqdm import tqdm
import os
import pickle
import numpy as np

#===================================================================================
# Parameters

input_data_dir = "data"
output_data_dir = "data/label"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

# labeling functions to apply
label_fns = [
    isa_pattern, 
    suchas_pattern, 
    including_pattern, 
    called_pattern, 
    especially_pattern,
    appo_pattern, 
    other_pattern, 
    knownas_pattern, 
    term_pos, 
    term_subset,
    kb_bio101_ds
]
# fix to extract name only
label_fn_names = [str(lf) for lf in label_fns]

splits = ['test', 'dev', 'train']

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    # apply the labellers
    applier = PandasLFApplier(lfs=label_fns)
    labels = {}
    for split in splits:
        
        with open(f"{input_data_dir}/{split}.pkl", 'rb') as fid:
            data = pickle.load(fid)
        
        print(f"Spacy preprocessing {split} data")
        docs = []
        for _, row in tqdm(list(data.iterrows())):
            docs.append(nlp(row.text))
        data['doc'] = docs
        
        print(f"Applying label functions to {split} data")
        labels[split] = applier.apply(df=data)
        np.savez_compressed(
            f"{output_data_dir}/{split}_label_fn_labels.npz", 
            data=labels[split], 
            label_fns=label_fn_names)
        lf_analysis = LFAnalysis(L=labels[split], lfs=label_fns).lf_summary()
        lf_analysis.to_csv(f"{output_data_dir}/{split}_label_analysis.csv")
        
        print(f"Majority vote labelling {split} data")
        majority_model = MajorityLabelVoter(cardinality=2)
        majority_labels = majority_model.predict(L=labels[split])
        np.savez_compressed(
            f"{output_data_dir}/{split}_majority_vote_labels.npz", 
            data=majority_labels, 
            label_fns=label_fn_names)
    
    print("Fitting LabelModel to Training Data")
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=labels['train'], n_epochs=500, log_freq=100, seed=123)
    
    for split in splits:
        print(f"Label model labelling {split} data")
        label_model_labels = label_model.predict_proba(labels[split])
        np.savez_compressed(
            f"{output_data_dir}/{split}_label_model_labels.npz", 
            data=label_model_labels, 
            label_fns=label_fn_names)
        
    
    
        
