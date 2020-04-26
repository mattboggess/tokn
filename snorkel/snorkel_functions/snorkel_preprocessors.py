from snorkel.preprocess import preprocessor


@preprocessor(memoize=True)
def get_text_between(cand):
    """
    Returns the text between two terms in a sentence.
    """
    term1_loc = cand.term1_location
    term2_loc = cand.term2_location
    if term1_loc[0] > term2_loc[0]:
        start = term2_loc[1]
        end = term1_loc[0]
        cand.term1_first = False
    else:
        start = term1_loc[1]
        end = term2_loc[0]
        cand.term1_first = True
        
    cand.text_between = ' '.join(cand.tokens[start:end])
    return cand

@preprocessor(memoize=True)
def get_left_text(cand):
    """
    Returns the text to the left of the first term in the sentence 
    """
    term1_loc = cand.term1_location
    term2_loc = cand.term2_location
    if term1_loc[0] > term2_loc[0]:
        lim = term2_loc[0]
    else:
        lim = term1_loc[0]
        
    cand.left_text = ' '.join(cand.tokens[:lim])
    return cand

@preprocessor(memoize=True)
def get_right_text(cand):
    """
    Returns the text to the right of the first term in the sentence 
    """
    term1_loc = cand.term1_location
    term2_loc = cand.term2_location
    if term1_loc[0] > term2_loc[0]:
        lim = term2_loc[0]
    else:
        lim = term1_loc[0]
        
    cand.left_text = ' '.join(cand.tokens[:lim])
    return cand