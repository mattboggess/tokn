# Converts Life Bio Raw HTML into sentences in pandas dataframe

import os
import re
import pandas as pd
from nltk import sent_tokenize

data_dir = '../data/raw_data/life_bio/life_bio_html'
sentence_regex = '<[pl]i*[ >].*?>.*?</[pl]i*>'
term_regex = '<span class="bolded-keyword">.*?</span>'

exclude_classes = [
    'fig-title',
    'h2a',
    'h5',
    'toc',
    'tab-title',
    'toc-section',
    'nextsectionautogen',
    'nextoverviewautogen',
    'nextoverviewtitle'
    'nextsectiontitle',
    'h7a',
    'h9',
    'h7a_new',
    'keyconcept',
    'h7title',
    'qa',
    'sidebar-division-title',
    'division-title'
]

def clean_sent(sent):
    sent = re.sub('<span class="sidebar-division-title".*?>.*?</span>', '', sent)
    sent = re.sub('<.*?>', '', sent)
    sent = sent.replace('*', '')
    sent = sent.replace('connect the concepts', '')
    sent = sent.replace('&#x2013;', '-')
    sent = sent.replace('&#8211;', '-')
    sent = sent.replace('&#201C;', '"')
    sent = sent.replace('&#201D;', '"')
    sent = sent.replace('#x2019;', "'")
    sent = sent.replace('&#8212;', '-')
    sent = sent.replace('&#8217;', "'")
    sent = sent.replace('&#8220;', '"')
    sent = sent.replace('&#8221;', '"')
    sent = sent.replace('&#0176;', '°')
    sent = sent.replace('&#8722;', '-')
    sent = sent.replace('&#0039;', '-')
    sent = sent.replace('&#0225;', 'á')
    
    sent = sent.replace('&lt;', '<')
    sent = sent.replace('&gt;', '>')
    sent = sent.strip()
    return sent

chs = sorted([ch for ch in os.listdir(data_dir) if ch.startswith('chapter')])
data = {
    'chapter': [],
    'section': [],
    'section_name': [],
    'sentence_number': [],
    'sentence': []
}

for ch in chs:
    ch_num = re.match('chapter(\d+)', ch).group(1) 
    sections = sorted([sect for sect in os.listdir(f"{data_dir}/{ch}") if sect.startswith('chapter')])
    for sect in sections:
        with open(f"{data_dir}/{ch}/{sect}", 'r') as fid:
            text = fid.read()
        text = text.replace('\n', ' ')
        
        section = re.match('chapter\d+-*(.*).html', sect).group(1)
        if not section:
            section = 0
        if section == 'summary':
            section = len(sections) - 1
        section_name = clean_sent(re.findall('<h1 class="concept-title">(.*?)</h1>', text)[0])
        
        sents = re.findall(sentence_regex, text)
        terms = re.findall(term_regex, text)
        new_sents = []
        for sent in sents:
            first_class = re.match('.*?class="(.*?)".*', sent)
            if first_class:
                first_class = first_class.group(1)
            else:
                first_class = ''
            if first_class in exclude_classes:
                if ch_num == '39':
                    print(sent)
                continue
            new_sents += sent_tokenize(clean_sent(sent))
        sents = new_sents
        sents = [s for s in sents if s != 'You should be able to:' and \
                                     s != "Apply what you've learned" and \
                                     len(s.split()) > 3 and \
                                     not s.startswith('End of') and \
                                     not s.startswith('Review')]
        data['sentence'] += sents
        data['sentence_number'] += list(range(1, len(sents) + 1))
        data['chapter'] += [ch_num] * len(sents) 
        data['section'] += [section] * len(sents) 
        data['section_name'] += [section_name] * len(sents) 
data = pd.DataFrame(data)
data.to_csv('../data/preprocessed/parsed_books/sentences_Life_Biology_parsed.csv', index=False)