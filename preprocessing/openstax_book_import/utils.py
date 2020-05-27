import re

yaml_path = './yaml_files/'
book_path = './book_files/'
sentences_path = './sentence_files/'
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
