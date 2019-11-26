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

def macro_recall(true, pred):
    return recall_score(true, pred, average="macro")

def compute_relation_metrics(wp_classifications):
    
    output = {}
    for relation in wp_classifications.keys():
        preds = []
        labels = []
        preds += [1] * len(wp_classifications[relation]["true_positives"])
        labels += [1] * len(wp_classifications[relation]["true_positives"])
        preds += [1] * len(wp_classifications[relation]["false_positives"])
        labels += [0] * len(wp_classifications[relation]["false_positives"])
        preds += [0] * len(wp_classifications[relation]["false_negatives"])
        labels += [1] * len(wp_classifications[relation]["false_negatives"])
        
        output[relation] = {}
        output[relation]["recall"] = recall_score(labels, preds)
        output[relation]["precision"] = precision_score(labels, preds)
        output[relation]["f1"] = f1_score(labels, preds)
    
    return output

def get_word_pair_classifications(predictions, target, word_pairs, relations):
    
    wp_classifications = {}
    for relation in relations:
        wp_classifications[relation] = {
            "false_positives": [], 
            "true_positives": [],
            "false_negatives": []
        }
    
    for pred, label, word_pair in zip(predictions, target, word_pairs):
        if pred == label:
            wp_classifications[relations[pred]]["true_positives"].append(word_pair)
        else:
            wp_classifications[relations[pred]]["false_positives"].append(word_pair)
            wp_classifications[relations[label]]["false_negatives"].append(word_pair)
    
    return wp_classifications
    