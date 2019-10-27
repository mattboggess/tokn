import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

# write out file with FP, TP, FN, TN for all classes
# training & validation & test

def accuracy(true, pred):
    return accuracy_score(true, pred)

def micro_f1(true, pred):
    return f1_score(true, pred, average="micro")

def macro_f1(true, pred):
    return f1_score(true, pred, average="macro")

def micro_precision(true, pred):
    return precision_score(true, pred, average="micro")

def macro_precision(true, pred):
    return precision_score(true, pred, average="macro")

def micro_recall(true, pred):
    return recall_score(true, pred, average="micro")

def macro_recall(true, pred):
    return recall_score(true, pred, average="macro")

