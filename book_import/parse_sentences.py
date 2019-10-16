# Read OpenStax textbook data and convert into tokenized sentences
# Author: drew

import pandas as pd
import re
import os
from ecosystem_importer import EcosystemImporter
from nltk.tokenize import sent_tokenize
from utils import *

### User params -- the yaml file from which we extract metadata
#yaml = 'Biology_2e_8d50a0af-948b-4204-a71d-4826cba765b8_15.45.yml'
yaml = 'College_Physics_with_Courseware_405335a3-7cff-4df2-a9ad-29062a4af261_7.53.yml'

### Compute some filename constants
subject_name = yaml.split('_')[0].lower()
yaml_file = yaml_path + yaml
book_file = book_path + '{}_book.csv'.format(subject_name)
final_output_file = sentences_path + 'final_{}_parsed.csv'.format(subject_name) 

### Step 1 -- read in the full book ###
df_book = None
if os.path.exists(book_file):
	print("Loading book from memory")
	df_book = pd.read_csv(book_file)
else:
	print("Can't find book . . . regenerating from yaml")
	es = EcosystemImporter()
	df_book = es.parse_yaml_file(yaml_file)
	df_book['content_clean'] = df_book['content'].apply(lambda x: es.format_cnxml(x))
	df_book.to_csv(book_file, index=None)


### Step 2 -- convert book into sentences ###
df_sentences = pd.DataFrame()
chapter_num = 0
section_num = 0
for ii in range(0, len(df_book)):
	content = df_book['content_clean'].iloc[ii]
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
	else:
		pass

	# Strip the name out of the content (first occurence)
	if not name.startswith('Key Terms'):
		m = re.search(name, content)
		if m:
			ending = m.span()[1]
			content = content[ending:].strip()

	sentences = sent_tokenize(content)
	sentences = [s.strip() for s in sentences]

	# Handle key terms
	if (len(sentences)==1):
		s = sentences[0]
		if (s.startswith('Key Terms')):
			sentences = proc_key_terms(df_book['content'].iloc[ii])

	# Handle weird number stuff in the end sections
	odd_ducks = ['Visual Connection Questions', 'Review Questions', 'Critical Thinking Questions']
	if name in odd_ducks:
		sentences = [re.sub('\d+  .', '', s) for s in sentences]
	
	# Take care of some extraneous weird stuff (single period sentences, etc)
	sentences = [s for s in sentences if len(s)>2]

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





