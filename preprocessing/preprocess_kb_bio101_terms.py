import spacy
import pandas as pd
from io import StringIO
from tqdm import tqdm

life_bio_data_dir = '../data/raw_data/life_bio/bio101_kb'
term_dir = '../data/preprocessed/terms'
kb_dir = '../data/preprocessed'

bio_concepts_file = f"{life_bio_data_dir}/kb_biology_concepts.txt"
lexicon_input_file = f"{life_bio_data_dir}/kb_lexicon.txt"
terms_output_file = f"{term_dir}/kb_bio101_terms.txt"
rel_terms_output_file = f"{term_dir}/kb_bio101_relations_exclude.txt"

# funky concepts
exclude_concepts = ['Aggregate', 'Center', 'Grouping-Activity', 'Normal', 'Person',
                    'Unstable-System', 'Sequence', 'Region']
# text representations of concepts that are too general and thus problematic for text matching
exclude_terms = ['object', 'aggregate', 'group', 'thing', 'region', 'center', 'response',
                 'series', 'unit', 'result', 'normal', 'divide', 'whole', 'someone', 'somebody',
                 'feature', 'class']


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
    #lexicon = lexicon.groupby('concept')['text'].apply(
    #    lambda x: list(set([t for t in x if t not in exclude_terms]))).reset_index()
    #lexicon = lexicon[lexicon.text.map(len) > 0]

    # spacy process terms to get lemmas
    #term_df = {'kb_concept': [], 'term_type': [], 'term': [], 'doc': []}
    terms = []
    relation_terms = set()
    seen_terms = set()
    print("Running text representations for each concept through Spacy NLP pipeline")
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
        
        # extract text representations for the concept
        #terms = list(lexicon.loc[lexicon.concept == concept, 'text'])[0]
        #terms = [t.replace('"', '').strip().replace('-', ' ') for t in terms]
        #terms = [t for t in terms if len(t) > 4]
        
        # add text of concept itself
        #if concept_text.lower() not in terms and concept_text.upper() not in terms:
        #    terms.append(concept_text)
            
        # remove exclude terms
        #terms = [t for t in terms if t not in exclude_terms]
        #if len(terms) == 0:
        #    continue
        
        
        # pull out event/entity label
        #term_type = concept_types.loc[concept_types.concept == concept, 'text'].iloc[0]
        
        # spacy preprocess 
        #docs = [nlp(term) for term in terms]
        #
        #term_df['kb_concept'] += [concept] * len(terms)
        #term_df['term_type'] += [term_type] * len(terms)
        #term_df['term'] += terms
        #term_df['doc'] += docs 
    
    with open(terms_output_file, 'w') as fid:
        fid.write('\n'.join(sorted(terms)))
    with open(rel_terms_output_file, 'w') as fid:
        fid.write('\n'.join(sorted(list(relation_terms))))
    
    #terms = pd.DataFrame(term_df)
    #terms['source'] = 'KB_Bio101'
    #terms.to_pickle(terms_output_file)