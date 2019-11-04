import json 
import numpy as np

ID2WORD = json.load(open(os.path.join("data/", 'id2word.json')))

def bag_of_words_featurizer(bag_of_words):
    """
    Featurizes bag of words into a vector based on word count
    bag_of_words is a list of lists of word ids
    """
    feat_len = len(ID2WORD)
    feat_vec = np.zeros(feat_len)
    for sentence in bag_of_words:
        for word_id in sentence:
            feat_vec[word_id] += 1
    return feat_vec