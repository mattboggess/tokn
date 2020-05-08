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
    
def get_term_classifications(predictions):
    relation_classes = ['OTHER', 'HYPONYM', 'HYPERNYM']

    term_class = {}
    for rel in relation_classes[1:]:
        term_class[rel] = {'false_positives': [], 'true_positives': [], 'false_negatives': []} 
        tp = list(predictions[(predictions.relation == rel) & (predictions.prediction == relation_classes.index(rel))].term_pair.unique())
        term_class[rel]['true_positives'] = tp
        fp = list(predictions[(predictions.relation != rel) & (predictions.prediction == relation_classes.index(rel))].term_pair.unique())
        term_class[rel]['false_positives'] = fp
        fn = list(predictions[(predictions.relation == rel) & (predictions.prediction != relation_classes.index(rel))].term_pair.unique())
        term_class[rel]['false_negatives'] = fn

    return term_class

def compute_term_metrics(term_class):
    metrics = {}
    tp = []
    fp = []
    fn = []
    for rel in term_class:
        metrics[rel] = {'precision': 0, 'recall': 0, 'f1': 0}
        tp += term_class[rel]['true_positives']
        fn += term_class[rel]['false_negatives']
        fp += term_class[rel]['false_positives']
        metrics[rel]['precision'] = len(term_class[rel]['true_positives']) / (len(term_class[rel]['false_positives']) + len(term_class[rel]['true_positives']))
        metrics[rel]['recall'] = len(term_class[rel]['true_positives']) / (len(term_class[rel]['false_negatives']) + len(term_class[rel]['true_positives']))
        metrics[rel]['f1'] = 2 * (metrics[rel]['precision'] * metrics[rel]['recall']) / (metrics[rel]['precision'] + metrics[rel]['recall'])

    metrics['overall'] = {}
    metrics['overall']['precision'] = len(set(tp)) / (len(set(tp)) + len(set(fp))) 
    metrics['overall']['recall'] = len(set(tp)) / (len(set(tp)) + len(set(fn))) 
    metrics['overall']['f1'] = 2 * (metrics['overall']['precision'] * metrics['overall']['recall']) / (metrics['overall']['precision'] + metrics['overall']['recall'])
    return metrics

