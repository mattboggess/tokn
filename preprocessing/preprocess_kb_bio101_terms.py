# Pulls out biology terms from a dump of the Bio101 knowledge base lexicon. 
#
# Author: Matthew Boggess
# Version: 4/26/20
#
# Data Source: Lexicon dump of Bio101 KB provided by Dr. Chaudhri 
#
# Description: For each biology concept in the Bio101 KB, enumerates all surface forms for each
# concept and saves these to a text file. Also saves out relations (volume, area, etc.) that are
# used to filter the final key term list.

#===================================================================================
# Libraries 

import spacy
import pandas as pd
from io import StringIO
from tqdm import tqdm

#===================================================================================
# Parameters 

life_bio_data_dir = '../data/raw_data/life_bio/bio101_kb'
term_dir = '../data/preprocessed/terms'
kb_dir = '../data/preprocessed'

bio_concepts_file = f"{life_bio_data_dir}/kb_biology_concepts.txt"
lexicon_input_file = f"{life_bio_data_dir}/kb_lexicon.txt"
terms_output_file = f"{term_dir}/kb_bio101_terms.txt"
rel_terms_output_file = f"{term_dir}/kb_bio101_relations_exclude.txt"

# concepts that are too general
exclude_concepts = ['Aggregate', 'Center', 'Grouping-Activity', 'Normal', 'Person',
                    'Unstable-System', 'Sequence', 'Region']

#===================================================================================

if __name__ == '__main__':
    
    seen_lemmas = set()
    terms = []
    
    # initialize Spacy NLP pipeline
    nlp = spacy.load('en_core_web_sm')
    
    print("Extracting Life Biology KB terms")
    with open(lexicon_input_file, "r") as f:
        lexicon = f.read()
    with open(bio_concepts_file, "r") as f:
        bio_concepts = set([t.strip() for t in f.readlines()])
    
    lexicon = pd.read_csv(StringIO(lexicon), sep="\s*\|\s*", header=None, engine='python',
                          names=['concept', 'relation', 'text', 'pos'])
    
    # separate out event/entity labels
    concept_types = lexicon.query("text in ['Entity', 'Event', 'Relation']")
    lexicon = lexicon.query("text not in ['Entity', 'Event', 'Relation']")
    
    # filter out too general upper ontology words, relation concepts, and 
    # concepts that only have representations that are too general 
    upper_ontology = lexicon[~lexicon.concept.isin(bio_concepts)]
    upper_ontology = set(upper_ontology.text)
    lexicon = lexicon[(lexicon.concept.isin(bio_concepts)) & (~lexicon.concept.isin(exclude_concepts))]

    terms = []
    relation_terms = set()
    seen_terms = set()
    for _, row in tqdm(list(lexicon.iterrows())):
        
        # pull out event/entity label
        term_type = concept_types.loc[concept_types.concept == row.concept, 'text'].iloc[0]
        if term_type == 'Relation':
            relation_terms.add(row.text.replace('"', '').strip().replace('-', ' '))
            continue
        
        term = row.text.replace('"', '').strip().replace('-', ' ')
        if term in upper_ontology:
            print(term)
        elif term not in seen_terms:
            terms.append(f"{term} -- {term_type.lower()} -- {row.concept}")
            seen_terms.add(term)
            
        concept_text = row.concept.lower().replace('-', ' ').strip()
        if concept_text in upper_ontology:
            print(concept_text)
        elif concept_text not in seen_terms:
            terms.append(f"{concept_text} -- {term_type.lower()} -- {row.concept}")
            seen_terms.add(concept_text)
        
    with open(terms_output_file, 'w') as fid:
        fid.write('\n'.join(sorted(terms)))
    with open(rel_terms_output_file, 'w') as fid:
        fid.write('\n'.join(sorted(list(relation_terms))))
