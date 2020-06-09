# Read OpenStax textbook data and convert into tokenized sentences
# Author: drew (modifications by Matt)

import pandas as pd
import re
import os
from ecosystem_importer import EcosystemImporter
from nltk.tokenize import sent_tokenize
import string

yaml_path = './yaml_files/'
book_path = '../../data/raw_data/openstax/book_files/'
sentences_path = '../../data/preprocessed/parsed_books/'
archive_url = 'https://archive.cnx.org/contents/{}'

def format_cnxml(text):
    clean = re.compile("<.*?>")
    clean_text = re.sub(clean, "", text)
    clean_text = clean_text.replace("\n", " ")
    clean_text = clean_text.replace("\\text{", " ")
    clean_text = clean_text.replace("}", " ")
    clean_text = clean_text.replace("≤", "<=")
    clean_text = clean_text.replace('˚', "")
    clean_text = clean_text.replace('‘', "'")
    clean_text = clean_text.replace('′', "'")
    clean_text = clean_text.replace('≥', ">=")
    clean_text = clean_text.replace('”', '"')
    clean_text = clean_text.replace('”', '"')
    clean_text = clean_text.replace('\xa0', ' ')
    clean_text = clean_text.replace('\u200b', ' ')
    clean_text = clean_text.replace('°', '')
    clean_text = clean_text.replace('’', "'")
    clean_text = clean_text.replace('↔', "<->")
    clean_text = clean_text.replace('ª', "")
    clean_text = clean_text.replace('“', '"')
    clean_text = clean_text.replace('º', '')
    clean_text = clean_text.replace('½', '1/2')
    clean_text = clean_text.replace('→', '->')
    clean_text = clean_text.replace('−', '-')
    return clean_text.strip()

def get_clean_sentences(content):
    sent = re.compile('(<p.*?>\n*.*?\n*</p>)|(<span class="os-caption">.*?</span>)')
    sections = re.findall(sent, content)
    sents = [format_cnxml(' '.join(s)) for s in sections]
    return ' '.join(sents)

def proc_key_terms(raw_cnxml):    
    term_regex_start = '<dt id="[fs\-id]*\d+">'
    term_regex_end = '<\/dt>'
    def_regex_start = '<dd id="fs-id[m,p]*\d*[a]*">'
    def_regex_end = '<\/dd>'
    # First let's check and make sure that we get the right number of matches for all regex
    Ts = len(re.findall(term_regex_start, raw_cnxml))
    Te = len(re.findall(term_regex_end, raw_cnxml))
    Ds = len(re.findall(def_regex_start, raw_cnxml))
    De = len(re.findall(def_regex_end, raw_cnxml))
    if (Ts==Te==Ds==De):
        # Do finditer for terms and defs -- get stuff in the middle and store in lists
        I_ts = [m.end() for m in re.finditer(term_regex_start, raw_cnxml)]
        I_te = [m.start() for m in re.finditer(term_regex_end, raw_cnxml)]
        I_ds = [m.end() for m in re.finditer(def_regex_start, raw_cnxml)]
        I_de = [m.start() for m in re.finditer(def_regex_end, raw_cnxml)]

        # Now get the term and def strings using the above lists
        terms = [raw_cnxml[I_ts[ii]:I_te[ii]] for ii in range(0, len(I_ts))]
        defs = [raw_cnxml[I_ds[ii]:I_de[ii]] for ii in range(0, len(I_ds))]

        # Weave into a thing and return (TODO clean the CNXML crap)
        output_str = [format_cnxml(t[0] + ": " + t[1]) for t in zip(terms, defs)]
        return output_str
    else:
        print("Something went wrong with the key term parsing . . . sorry about that.")
        return False
    
def proc_index(raw_cnxml):
    term_regex = '<span group-by=.*? class="os-term">.*?</span>'
    section_regex = '<span class="os-term-section">.*?</span>'
    terms = re.findall(term_regex, raw_cnxml)
    secs = re.findall(section_regex, raw_cnxml)
    
    output_str = [format_cnxml(t[0] + ": " + t[1]) for t in zip(terms, secs)]
    return output_str

def proc_key_terms_microbio(raw_cnxml):
    term_regex = '<li><strong>.*?</strong>'
    def_regex = '</strong>.*?</li>'
    terms = re.findall(term_regex, raw_cnxml)
    defs = re.findall(def_regex, raw_cnxml)
    
    output_str = [format_cnxml(t[0] + ": " + t[1]) for t in zip(terms, defs)]
    return output_str

def clean_paren_stuff(x):
    x = re.sub("^\(.*redit.*\)", "", x).strip()
    if ("redit:" in x):  #Captures Credit and credit
        x = ""
    if (len(x)>0):
        if (x[0]=='('):
            x = x[1:]
    return x

### Load up the cnx id lookup table
df_id = pd.read_csv('../../data/raw_data/openstax/book_cnxid.csv')

### User params -- the yaml file from which we extract metadata
book_title = 'Anatomy and Physiology' # This name needs to match a title in df_id
cnx_id = df_id[df_id['title']==book_title].cnx_id.iloc[0]

### Compute some filename constants
translator = str.maketrans('', '', string.punctuation)
subject_name = book_title.translate(translator).replace(' ', '_')
book_file = book_path + '{}_book.csv'.format(subject_name)
final_output_file = sentences_path + 'sentences_{}_parsed.csv'.format(subject_name)

### Step 1 -- read in the full book ###
df_book = None
if os.path.exists(book_file):
    print("Loading book from memory")
    df_book = pd.read_csv(book_file)
else:
    print("Can't find book . . . regenerating from yaml")
    es = EcosystemImporter()
    df_book = es.get_book_content(archive_url, cnx_id)
    df_book['content_clean'] = df_book['content'].apply(lambda x: es.format_cnxml(x))
    df_book.to_csv(book_file, index=None)


### Step 2 -- convert book into sentences ###
df_sentences = pd.DataFrame()
chapter_num = int(0)
section_num = int(0)
for ii in range(0, len(df_book)):
    content = df_book['content_clean'].iloc[ii]
    sent_content = get_clean_sentences(df_book['content'].iloc[ii])
    if (content.startswith("Chapter Outline")):
        chapter_num = chapter_num + 1
        section_num = 1
    else:
        section_num += 1

    # Get section/chapter name
    ch_name = re.findall('<h3 class="os-title">([\w\s,:]+)<\/h3>', df_book['content'].iloc[ii])
    sec_name = re.findall('<span class="os-text">([\w\s,:]+)<\/span>', df_book['content'].iloc[ii])
    name = ''
    if (len(ch_name)>0):
        name = ch_name[0]
    elif (len(sec_name)>0):
        name = sec_name[0]
    elif content.startswith('Index'):
        name = 'Index'
        print(content)
    else:
        pass

    # Strip the name out of the content (first occurence)
    #if not name.startswith('Key Terms'):
    #    m = re.search(name, content)
    #    if m:
    #        ending = m.span()[1]
    #        content = content[ending:].strip()

    sentences = sent_tokenize(sent_content)
    sentences = [s.strip() for s in sentences]

    if (name.startswith('Key Terms')):
        sentences = proc_key_terms(df_book['content'].iloc[ii])
    # special handling for microbiology key terms
    if 'Glossary' in name:
        name = 'Key Terms'
        sentences = proc_key_terms_microbio(df_book['content'].iloc[ii])
    
    # parse out index
    if name == 'Index':
        sentences = proc_index(df_book['content'].iloc[ii])

    # Handle weird number stuff in the end sections
    odd_ducks = ['Visual Connection Questions', 'Review Questions', 'Critical Thinking Questions']
    if name in odd_ducks:
        sentences = [re.sub('\d+  .', '', s) for s in sentences]

    # Take care of some extraneous weird stuff (single period sentences, etc)
    sentences = [s for s in sentences if len(s)>2 and s != '.']
    

    page_id = df_book['page_id'].iloc[ii]
    Ns = len(sentences)
    dft = pd.DataFrame(
        {
            'page_id': [page_id]*Ns,
            'chapter': [chapter_num]*Ns,
            'section': [section_num]*Ns,
            'section_name': [name]*Ns,
            'sentence': sentences,
            'sentence_number': range(1, Ns+1),
        }
    )
    df_sentences = df_sentences.append(dft)


### Do some final cleanup ###
df_sentences['sentence'] = df_sentences['sentence'].apply(lambda x: clean_paren_stuff(x))
df_sentences = df_sentences[df_sentences['sentence']!=""]
df_sentences = df_sentences[['page_id', 'chapter', 'section', 'section_name', 'sentence_number', 'sentence']]

### Write to disk ###
df_sentences.to_csv(final_output_file, index=None)





