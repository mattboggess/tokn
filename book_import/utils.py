import re

yaml_path = './yaml_files/'
book_path = './book_files/'
sentences_path = './sentence_files/'

def proc_key_terms(raw_cnxml):	
	term_regex_start = '<dt id="\\d+">'
	term_regex_end = '<\/dt>'
	def_regex_start = '<dd id="fs-id[m,p]*\d*">'
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
		output_str = [t[0] + ": " + t[1] for t in zip(terms, defs)]
		return output_str
	else:
		return False

def clean_paren_stuff(x):
	x = re.sub("^\(.*redit.*\)", "", x).strip()
	if ("redit:" in x):  #Captures Credit and credit
		x = ""
	if (len(x)>0):
		if (x[0]=='('):
			x = x[1:]
	return x
