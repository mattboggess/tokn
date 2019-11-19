import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

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

def get_word_pair_classifications(predictions, target, word_pairs, relations):
    
    wp_classifications = {}
    for relation in relations:
        wp_classifications[relation] = {
            "false_positives": [], 
            "true_positives": []
        }
    
    for pred, label, word_pair in zip(predictions, target, word_pairs):
        if pred == label:
            wp_classifications[relations[pred]]["true_positives"].append(word_pair)
        else:
            wp_classifications[relations[pred]]["false_positives"].append(word_pair)
    
    return wp_classifications
    