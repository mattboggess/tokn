import spacy
import pandas as pd
from io import StringIO
import os
from tqdm import tqdm

# text representations of concepts that are too general and thus problematic for text matching
exclude_terms = ['object', 'aggregate', 'group', 'thing', 'region', 'center', 'response',
                 'series', 'unit', 'result', 'normal', 'divide', 'whole', 'someone', 'somebody',
                 'feature', 'class']
# data directories
exclude_term_file
raw_data_dir = "../data/raw_data/hand_labelled"
output_dir = "../data/preprocessed/terms"
    
## Important Enumerations 

# textbook sections/chapters to be processed
textbook_sections = [
    'openstax_bio2e_section10-2_hand_labelled',
    'openstax_bio2e_section10-4_hand_labelled',
    'openstax_bio2e_section4-2_hand_labelled',
    'life_bio_ch39_hand_labelled'
]

#===================================================================================

if __name__ == "__main__":
    
    for i, section in enumerate(textbook_sections):
        
        nlp = spacy.load('en_core_web_sm')

        print(f"Processing {section} textbook section: Section {i + 1}/{len(textbook_sections)}")
            
        term_df = {'term_type': [], 'term': [], 'doc': []}
        with open(f"{raw_data_dir}/{section}_terms.txt", 'r') as fid:
            key_terms = fid.readlines()
            for key_term in tqdm(key_terms):
                kt, tt = key_term.split('--')
                doc = nlp(kt.strip())
                kt = ' '.join([t.lemma_.lower() for t in doc])
                if len(kt) and kt not in term_df['term']:
                    term_df['term'].append(kt)
                    term_df['term_type'].append(tt.strip())
                    term_df['doc'].append(doc)
            
        terms = pd.DataFrame(term_df)
        terms['source'] = section
        terms.to_pickle(f"{output_dir}/{section}_terms.pkl")
                    
