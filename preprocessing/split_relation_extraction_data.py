# Splits sentences tagged with term pairs into a train, dev, and test split for relation extraction.
#
# Author: Matthew Boggess
# Version: 5/28/20
#
# Data Source: Sentence, term pair grouped dataframes from generate_term_pairs.py
#
# Description: Combines Life Biology and OpenStax biology tagged sentences/term pairs and split them 
# into a train, dev, and test set removing overlap between the train and dev/test set.

#===================================================================================
# Libraries 

import pandas as pd

#===================================================================================
# Parameters

life_bio_file = "../data/preprocessed/term_pair_sentences/Life_Biology_term_pairs.pkl"
openstax_bio_file = "../data/preprocessed/term_pair_sentences/Biology_2e_term_pairs.pkl"
output_dir = "../data/relation_extraction"

# Dev and test texbtooks and chapters
dev_chapters = [
    ('Biology_2e', 4), 
    ('Biology_2e', 10), 
    ('Life_Biology', 5), 
    ('Life_Biology', 11)
]
test_chapters = [('Life_Biology', 39)]

# include terms labelled in both directions or not 
one_direction = True

#===================================================================================
# Helper Functions 

def filter_long(df):
    """
    Filter out invalid sentences that are too long and likely not real sentences (more than 12 terms)
    """ 
    invalid = \
        df \
        .groupby(['sentence']) \
        .chapter \
        .agg(count='count') \
        .reset_index() \
        .query('count > 100')
    df = df[~df.sentence.isin(invalid.sentence)]
    return df

#===================================================================================

if __name__ == '__main__':
    
    life = pd.read_pickle(life_bio_file)
    openstax = pd.read_pickle(openstax_bio_file)
    data = pd.concat([life, openstax])
    
    # filter out reverse directions so term1 is always first
    if one_direction:
        data = data[data.apply(lambda x: x['term1_location'][0] < x['term2_location'][0], axis=1)]
        
    # filter out invalid too long sentences (intro sentences from OpenStax) 
    data = data[~data.sentence.str.contains('By the end of this section')]
    data = filter_long(data)
    
    # filter out splits based on textbok and chapter
    data['filter_col'] = list(zip(data['textbook'], data['chapter']))
    dev = data[data.filter_col.isin(dev_chapters)]
    test = data[data.filter_col.isin(test_chapters)]
    train = data[(~data.filter_col.isin(dev_chapters)) & (~data.filter_col.isin(test_chapters))]
    
    # remove term pair overlap with dev and test from train
    train['term_pair'] = list(zip(train['term1'], train['term2']))
    dev['term_pair'] = list(zip(dev['term1'], dev['term2']))
    test['term_pair'] = list(zip(test['term1'], test['term2']))
    train = train[~train.term_pair.isin(dev.term_pair)]
    train = train[~train.term_pair.isin(test.term_pair)]
    
    # write splits to file  
    train.drop(['filter_col'], axis=1).to_pickle(f"{output_dir}/train.pkl")
    dev.drop(['filter_col'], axis=1).to_pickle(f"{output_dir}/dev.pkl")
    test.drop(['filter_col'], axis=1).to_pickle(f"{output_dir}/test.pkl")