import torch
import spacy
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
import sys
import os
import warnings
import numpy as np

def accuracy(true, pred):
    return accuracy_score(true, pred)

def token_macro_f1(true, pred):
    return f1_score(true, pred, labels=[1, 2, 3, 4], average='macro')

def token_macro_precision(true, pred):
    return precision_score(true, pred, labels=[1, 2, 3, 4], average='macro')

def token_macro_recall(true, pred):
    return recall_score(true, pred, labels=[1, 2, 3, 4], average='macro')

def token_micro_f1(true, pred):
    return f1_score(true, pred, labels=[1, 2, 3, 4], average='micro')

def token_micro_precision(true, pred):
    return precision_score(true, pred, labels=[1, 2, 3, 4], average='micro')

def token_micro_recall(true, pred):
    return recall_score(true, pred, labels=[1, 2, 3, 4], average='micro')

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

def extract_terms_from_sentence(model_probs, doc, tags, correct_phrases=False):
    """
    Extracts a list of terms and confidence scores for the model given a list of model predictions and a Spacy
    dependency parse for a sentence.
    
    Parameters
    ----------
    model_probs: num_tokens x num_tags array 
        Array of model probabilities across all tags for each token in the sentence 
    doc: Spacy Doc 
        Spacy preprocessed sentence
    tags: list of str
        List of BIOES tags used
    correct_phrases: boolean
        Indicator as to whether it should try to join neighboring singleton predictions together as the model has
        a tendency to split phrases up into multiple successive singletons
    
    Returns
    -------
    terms: list of str 
        List of terms predicted by the model for the sentence in lemmatized form 
    term_probs: list of float 
        List of confidence probabilities for each of the predicted terms 
    """
    i = 0
    num_tokens = len(model_probs)
    terms = []
    term_probs = []
    while i < num_tokens:
        tag = model_probs[i].argmax()
        tmp_term = []
        tmp_prob = []
        
        # continue if not start of term
        if tags[tag] in ['O', 'I', 'E']:
            i += 1
        
        # extract singleton terms
        elif tags[tag] == 'S':
            tmp_term.append(doc[i].lemma_.lower())
            tmp_prob.append(model_probs[i].max().item())
            
            # join together singular terms to correct for split phrases 
            if correct_phrases:
                if not doc[i].tag_.endswith('VB'):
                    while i < num_tokens:
                        i += 1
                        tag = model_probs[i].argmax()
                        if tags[tag] == 'S' and not doc[i].tag_.endswith('VB'):
                            tmp_term.append(doc[i].lemma_.lower())
                            tmp_prob.append(model_probs[i].max().item())
                        else:
                            break
                else:
                    i += 1
            else:
                i += 1
                    
        # extract term phrases
        elif tags[tag] == 'B':
            tmp_term.append(doc[i].lemma_.lower())
            tmp_prob.append(model_probs[i].max().item())
            i += 1
            while i < num_tokens:
                tag = model_probs[i].argmax()
                # erase if invalid stop of phrase
                if tags[tag] in ['O', 'S', 'B']:
                    tmp_term = []
                    tmp_prob = []
                    break
                # continue on if we are still in phrase
                elif tags[tag] == 'I':
                    tmp_term.append(doc[i].lemma_.lower())
                    tmp_prob.append(model_probs[i].max().item())
                    i += 1
                # break if we successfully hit end of phrase
                elif tags[tag] == 'E':
                    tmp_term.append(doc[i].lemma_.lower())
                    tmp_prob.append(model_probs[i].max().item())
                    i += 1
                    break
        
        if len(tmp_term):
            terms.append(' '.join(tmp_term).replace(' - ', ' ').strip())
            term_probs.append(np.mean(tmp_prob))
            
    return terms, term_probs
        

def get_term_predictions(model_output, mask, docs, tags):
    """
    Extracts out predicted terms and confidence scores for a batch of sentences
    
    Parameters
    ----------
    model_output: batch_size x num_tokens x num_tags array 
        Array of model log probabilities across all tags for each token in the sentence 
    mask: batch_size x num_tokens
        Mask array denoting which tokens actually correspond to original tokens in the sentence 
    docs: list of Spacy Doc 
        List of Spacy preprocessed sentences
    tags: list of str
        List of BIOES tags used
    
    Returns
    -------
    terms: list of list str 
        List of terms predicted by the model for each sentence in lemmatized form 
    term_probs: list of list of float 
        List of confidence probabilities for each of the predicted terms in each sentence
    """
    batch_size = model_output.shape[0]
    model_probs = torch.exp(model_output)
    terms = []
    term_probs = []
    for i in range(batch_size):
        mk = mask[i, :] 
        mp = model_probs[i, mk == 1, :]
        tmp = extract_terms_from_sentence(mp, docs[i], tags)
        terms.append(tmp[0])
        term_probs.append(tmp[1])
    
    return terms, term_probs

def compute_term_categories(term_data):
    """
    Categorizes set of predicted terms into false positives, true positives, and false negatives.
    
    Input is a daframe with a column of labelled terms, predicted terms, and predicted term probabilities
    for each sentence.
    
    Output is a dictionary with false positives, false negatives, and true positives.
    """ 
    
    # collect labelled terms
    terms = set()
    for t in term_data.terms:
        terms = terms | t
    terms = set([t.strip() for t in terms])
        
    # combine all occurences of predicted terms to get single confidence scores
    tmp_terms = [t for ts in term_data.predicted_terms for t in ts]
    tmp_probs = [p for probs in term_data.term_probs for p in probs]
    predicted_terms = {}
    for term, prob in zip(tmp_terms, tmp_probs):
        if term not in predicted_terms:
            predicted_terms[term] = [prob]
        else:
            predicted_terms[term].append(prob)
    for t in predicted_terms:
        predicted_terms[t] = np.mean(predicted_terms[t])
                 
    categories = ["false_positives", "false_negatives", "true_positives"]
    output = {category: [] for category in categories}
    for category in categories:
        if category == "true_positives":
            category_terms = terms.intersection(set(predicted_terms.keys()))
        elif category == "false_positives":
            category_terms = set(predicted_terms.keys()).difference(terms)
        elif category == "false_negatives":
            category_terms = terms.difference(set(predicted_terms.keys()))
        
        for ct in category_terms:
            if ct in predicted_terms:
                output[category].append([ct, predicted_terms[ct]])
            else:
                output[category].append([ct, 0])
        
        output[category] = sorted(output[category], key = lambda x: (-x[1], x[0]))

    return output
        