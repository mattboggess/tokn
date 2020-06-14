import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def accuracy(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return accuracy_score(true, pred)

def micro_f1(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return f1_score(true, pred, labels=[1, 2, 3, 4, 5], average="micro")

def macro_f1(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return f1_score(true, pred, labels=[1, 2, 3, 4, 5],  average="macro")

def micro_precision(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return precision_score(true, pred, labels=[1, 2, 3, 4, 5], average="micro")

def macro_precision(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return precision_score(true, pred, labels=[1, 2, 3, 4, 5], average="macro")

def micro_recall(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return recall_score(true, pred, labels=[1, 2, 3, 4, 5], average="micro")

def macro_recall(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return recall_score(true, pred, labels=[1, 2, 3, 4, 5], average="macro")

def get_term_pair_predictions(instance_data):
    """
    Takes a dataframe with true labels and predictions per sentence and reduces them down to
    predictions per term pair collapsing across sentences.
    """
    instance_data['kb_label'] = instance_data.kb_label.replace('ABSTAIN', 'OTHER')
    instance_data['label_fn_label'] = instance_data.label_fn_label.replace('ABSTAIN', 'OTHER')
    
    # get unique gold labels for term pairs
    gl = instance_data[instance_data.hard_label_class != 'OTHER'][['term_pair', 'hard_label_class']] \
       .rename(columns={'hard_label_class': 'true_label'}) \
       .drop_duplicates()
    
    # get unique kb labels for term pairs
    kb = instance_data[instance_data.kb_label != 'OTHER'][['term_pair', 'kb_label']] \
       .drop_duplicates()
    
     # get unique lf labels for term pairs
    lf = instance_data[instance_data.label_fn_label != 'OTHER'][['term_pair', 'label_fn_label']] \
       .groupby(['term_pair', 'label_fn_label']) \
       .size() \
       .reset_index(name='count') \
       .sort_values(['count'], ascending=False) \
       .groupby('term_pair') \
       .first() \
       .reset_index() \
       .drop(['count'], axis=1)
    
    # get predicted labels for term pairs 
    pred = instance_data[['term_pair', 'predicted_relation', 'prediction_confidence']] \
       .rename(columns={'predicted_relation': 'predicted_label'}) \
       .groupby(['term_pair', 'predicted_label']) \
       .agg(num_predicted = ('prediction_confidence', 'count'), confidence=('prediction_confidence', 'mean')) \
       .reset_index()
    # only take the most likely predicted label for each term pair excluding OTHER (most frequently*most confidently)
    pred['score'] = pred['num_predicted'] * pred['confidence']
    pred['other_flag'] = pred['predicted_label'] != 'OTHER'
    pred = pred \
       .sort_values(['other_flag', 'score'], ascending=False) \
       .groupby('term_pair') \
       .first() \
       .reset_index() \
       .drop(['score', 'other_flag'], axis=1)
    
    # merge all labels together
    data = pd.merge(gl, pred, how='outer', on='term_pair')
    data = pd.merge(data, kb, how='outer', on='term_pair')
    data = pd.merge(data, lf, how='outer', on='term_pair')
    
    data['label_fn_label'] = data['label_fn_label'].fillna('OTHER')
    data['kb_label'] = data['kb_label'].fillna('OTHER')
    data['true_label'] = data['true_label'].fillna('OTHER')
    
    # remove true negatives
    tn = (data.kb_label == 'OTHER') & (data.label_fn_label == 'OTHER') & (data.predicted_label == 'OTHER') & (data.true_label == 'OTHER')
    return data[~tn]
    