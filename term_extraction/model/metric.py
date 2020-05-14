import torch
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
import sys
import os
import warnings
import numpy as np

def sentence_accuracy(true, pred):
    return accuracy_score(true, pred)

def sentence_f1(true, pred):
    return f1_score(true, pred)

def sentence_precision(true, pred):
    return precision_score(true, pred)

def sentence_recall(true, pred):
    return recall_score(true, pred)

def term_precision(term_classifications):
    num = len(term_classifications["true_positives"])
    denom = len(term_classifications["true_positives"]) + len(term_classifications["false_positives"])
    if denom == 0:
        return 0
    return num / denom 
    
def term_recall(term_classifications):
    num = len(term_classifications["true_positives"])
    denom = len(term_classifications["true_positives"]) + len(term_classifications["false_negatives"])
    if denom == 0:
        return 0
    return num / denom 
    
def term_f1(term_classifications):
    precision = term_precision(term_classifications)
    recall = term_recall(term_classifications)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def get_term_predictions(pred, target, bert_mask, sentences, tags, pred_probs=None):
    """
    Extracts predicted terms and term phrases from a set of tags for a sentence. 
    """
    # flatten batch dimension
    pred = pred.view(pred.shape[0] * pred.shape[1])
    pred_probs = pred_probs.view(pred_probs.shape[0] * pred_probs.shape[1]) if pred_probs is not None else pred_probs
    target = target.view(target.shape[0] * target.shape[1])
    bert_mask = bert_mask.view(bert_mask.shape[0] * bert_mask.shape[1])
    sentences = [s for sent in sentences for s in sent]
    
    # filter out extra bert tokens
    keep_ix = bert_mask == 1
    pred = pred[keep_ix]
    if pred_probs is not None:
        pred_probs = pred_probs[keep_ix]
    target = target[keep_ix]
    
    term_preds = []
    term_target = []
    terms = {}
    terms_probs = {}

    i = 0
    while i < len(target):
        pred_tag = pred[i].item()
        pred_prob = pred_probs[i].item() if pred_probs is not None else 0
        target_tag = target[i].item()
        token = sentences[i]
        
        # singleton term
        if tags[target_tag] == "S":
            term_target.append(1)
            if tags[pred_tag] == "S":
                term_preds.append(1)
                if token in terms:
                    terms[token] += 1
                    terms_probs[token].append(pred_prob)
                else:
                    terms[token] = 1
                    terms_probs[token] = []
                    terms_probs[token].append(pred_prob)
            else:
                term_preds.append(0)
                
        # non-key term
        elif tags[target_tag] == "O":
            term_target.append(0)
            if tags[pred_tag] == "O":
                term_preds.append(0)
            else:
                term_preds.append(1)
                if token in terms:
                    terms[token] += 1
                    terms_probs[token].append(pred_prob)
                else:
                    terms[token] = 1
                    terms_probs[token] = []
                    terms_probs[token].append(pred_prob)
        
        # key phrase
        elif tags[target_tag] == "B":
            term_target.append(1)
            label = [target_tag]
            predicted_label = [pred_tag]
            token = [token]
            predicted_prob = [pred_prob]
            while tags[target[i]] != "E" and i < len(target) - 1:
                i += 1
                label.append(target[i].item())
                predicted_label.append(pred[i].item())
                if pred_probs is not None:
                    predicted_prob.append(pred_probs[i].item())
                token.append(sentences[i])
            if " ".join([str(l) for l in label]) == " ".join([str(l) for l in predicted_label]):
                token = " ".join(token)
                term_preds.append(1)
                if token in terms:
                    terms[token] += 1
                    terms_probs[token].append(np.mean(np.array(predicted_prob)))
                else:
                    terms_probs[token] = []
                    terms_probs[token].append(np.mean(np.array(predicted_prob)))
                    terms[token] = 1
                
            else:
                term_preds.append(0)
        else:
            pass
        
        i += 1
    for token in terms_probs:
        terms_probs[token] = np.amax(np.array(terms_probs[token]))
    return {"prediction": term_preds, "target": term_target, "predicted_terms": terms, "probability_terms": terms_probs}


def compute_term_categories(terms, predicted_terms):
    """
    Categorizes set of predicted terms into false positives, true positives, and false negatives.
    """ 
    warnings.filterwarnings('ignore')
    sys.stdout = open(os.devnull, "w")
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    sys.stdout = sys.__stdout__
    
    # preprocess predicted_terms
    predicted_terms_spacy = Counter()
    for term in predicted_terms:
        term_lemma = " ".join([t.lemma_ for t in nlp(term)])
        predicted_terms_spacy.update({term_lemma: predicted_terms[term]})
    
    categories = ["false_positives", "false_negatives", "true_positives"]
    output = {category: {} for category in categories}
    for category in categories:
        if category == "true_positives":
            category_terms = set(terms.keys()).intersection(set(predicted_terms_spacy.keys()))
        elif category == "false_positives":
            category_terms = set(predicted_terms_spacy.keys()).difference(set(terms.keys()))
        elif category == "false_negatives":
            category_terms = set(terms.keys()).difference(set(predicted_terms_spacy.keys()))
        
        for ct in category_terms:
            if category == "true_positives":
                output[category][ct] = {"present": terms[ct],
                                        "predicted": predicted_terms_spacy[ct]}
            elif category == "false_positives":
                output[category][ct] = {"present": 0,
                                        "predicted": predicted_terms_spacy[ct]}
            elif category == "false_negatives":
                output[category][ct] = {"present": terms[ct],
                                        "predicted": 0}

    return output

