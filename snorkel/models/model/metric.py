import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def accuracy(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return accuracy_score(true, pred)

def micro_f1(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return f1_score(true, pred, labels=[1, 2], average="micro")

def macro_f1(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return f1_score(true, pred, labels=[1, 2],  average="macro")

def binary_f1(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return f1_score(true, pred, average="binary")

def micro_precision(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return precision_score(true, pred, labels=[1, 2], average="micro")

def macro_precision(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return precision_score(true, pred, labels=[1, 2], average="macro")

def binary_precision(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return precision_score(true, pred, average="binary")

def micro_recall(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return recall_score(true, pred, labels=[1, 2], average="micro")

def macro_recall(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return recall_score(true, pred, labels=[1, 2], average="macro")

def binary_recall(true, pred):
    pred = np.concatenate(pred).argmax(axis=-1).squeeze() 
    return recall_score(true, pred, average="binary")

def binary_roc_auc(true, scores):
    scores = np.stack(scores).squeeze()[:, 1]
    return roc_auc_score(true, scores)

def macro_roc_auc(true, scores):
    scores = np.stack(scores).squeeze()[:, 1]
    return roc_auc_score(true, scores, average="macro")

def macro_avg_precision(true, scores):
    scores = np.stack(scores).squeeze()[:, 1]
    return average_precision_score(true, scores, average="macro")

def micro_avg_precision(true, scores):
    scores = np.stack(scores).squeeze()[:, 1]
    return average_precision_score(true, scores, average="micro")
    
