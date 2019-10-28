def bag_of_words_featurizer(term_pair, feature_counter):
    sentences = term_pair['sentences']
    #sentences.find("<e1>") and indices to get middle
    for word in ex.middle.split(' '):
        feature_counter[word] += 1
    return feature_counter
