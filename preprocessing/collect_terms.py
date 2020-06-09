# Collects various term lists together into a single term list dataframe and Spacy preprocesses the terms. 
#
# Author: Matthew Boggess
# Version: 6/08/20
#
# Data Source: Output terms of preprocess_textbooks.py, preprocess_kb_bio101_terms.py, and any hand-labelled terms.
#
#===================================================================================
# Libraries 

import os
import pandas as pd
import re
from tqdm import tqdm
import spacy

#===================================================================================
# Parameters 

terms_dir = "../data/preprocessed/terms"

rel_terms_exclude_file = f"{terms_dir}/kb_bio101_relations_exclude.txt"
custom_terms_exclude_file = f"{terms_dir}/exclude_terms.txt"

domain_mapping = {
    'Biology_2e': 'biology',
    'kb_bio101': 'biology',
    'openstax_bio2e_section4-2_hand_labelled': 'biology',
    'openstax_bio2e_section10-2_hand_labelled': 'biology',
    'openstax_bio2e_section10-4_hand_labelled': 'biology',
    'life_bio_ch39_hand_labelled': 'biology',
    'Chemistry_2e': 'chemistry',
    'Astronomy': 'astronomy',
    'Anatomy_and_Physiology': 'anatomy_and_physiology',
    'Microbiology': 'microbiology',
    'Psychology': 'psychology',
    'University_Physics_Volume_1': 'physics',
    'University_Physics_Volume_2': 'physics',
    'University_Physics_Volume_3': 'physics'
}

#===================================================================================

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    
    exclude_terms = []
    with open(rel_terms_exclude_file, 'r') as fid:
        rel_terms = fid.readlines()
    exclude_terms += rel_terms
    exclude_terms = set([e.strip() for e in exclude_terms])
    
    term_df = {
        'term': [],
        'type': [],
        'source': [],
        'domain': [],
        'kb_concept': [],
        'doc': []
    }
    
    files = os.listdir(terms_dir)

    for i, file in enumerate(files):
        source = re.match('(.*)_terms.txt', file)
        if not source:
            continue
        source = source.group(1)
        print(source)
        
        domain = domain_mapping[source]
        
        with open(f"{terms_dir}/{file}", 'r') as fid:
            terms = fid.readlines()
        
        for t in tqdm(terms):
            tmp = t.split(' -- ')
            if len(tmp) == 3:
                term, ttype, concept = tmp
            elif len(tmp) == 2:
                term, ttype = tmp
                concept = 'N/A'
            else:
                term = tmp[0]
                concept = 'N/A'
                ttype = 'N/A'
                
            term = term.strip()
            ttype = ttype.replace('\n', '').strip()
            concept = concept.strip()
            
            doc = nlp(term) 
            lemma = ' '.join([tok.lemma_.lower() if tok.lemma_.lower() != '-pron-' else tok.text.lower() 
                              for tok in doc])
            lemma = lemma.strip()
            
            if lemma in exclude_terms:
                continue
            
            term_df['term'].append(lemma)
            term_df['type'].append(ttype)
            term_df['source'].append(source)
            term_df['domain'].append(domain)
            term_df['kb_concept'].append(concept)
            term_df['doc'].append(doc)
            
    term_df = pd.DataFrame(term_df)
    term_df = term_df.drop_duplicates(['term', 'source'])
    term_df = term_df.sort_values(['domain', 'source', 'term'])
    term_df.to_pickle(f"{terms_dir}/processed_terms.pkl")
    term_df[['term', 'type', 'source', 'domain', 'kb_concept']].to_csv(f"{terms_dir}/processed_terms.csv", index=False)
            