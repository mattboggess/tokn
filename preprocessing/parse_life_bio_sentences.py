# Parses an html version of life biology into a separate csv file with one row per cleaned sentence.
#
# Author: Matthew Boggess
# Version: 6/08/20
#
# Data Source: Raw html of Life Biology in raw data directory.
#
#===================================================================================
# Libraries 

import os
import re
import pandas as pd
from nltk import sent_tokenize

#===================================================================================
# Parameters

input_dir = "../data/raw_data/life_bio/life_bio_html"
output_dir = "../data/preprocessed/parsed_books/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# html regex for valid sentences
sentence_regex = '<[pl]i*[ >].*?>.*?</[pl]i*>'

# classes that indicate text that aren't valid sentences
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

#===================================================================================
# Helper Functions

def clean_sent(sent):
    """Removes html formatting and cleans up parsing errors for sentences."""
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

#===================================================================================

if __name__ == '__main__':
    
    chs = sorted([ch for ch in os.listdir(input_dir) if ch.startswith('chapter')])
    data = {
        'chapter': [],
        'section': [],
        'section_name': [],
        'sentence_number': [],
        'sentence': []
    }

    for ch in chs:
        ch_num = re.match('chapter(\d+)', ch).group(1) 
        sections = sorted([sect for sect in os.listdir(f"{input_dir}/{ch}") if sect.startswith('chapter')])
        for sect in sections:
            with open(f"{input_dir}/{ch}/{sect}", 'r') as fid:
                text = fid.read()
            text = text.replace('\n', ' ')

            section = re.match('chapter\d+-*(.*).html', sect).group(1)
            if not section:
                section = 0
            if section == 'summary':
                section = len(sections) - 1
            section_name = clean_sent(re.findall('<h1 class="concept-title">(.*?)</h1>', text)[0])

            sents = re.findall(sentence_regex, text)
            new_sents = []
            for sent in sents:
                first_class = re.match('.*?class="(.*?)".*', sent)
                if first_class:
                    first_class = first_class.group(1)
                else:
                    first_class = ''
                if first_class in exclude_classes:
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
    data.to_csv(f"{output_dir}/sentences_Life_Biology_parsed.csv", index=False)