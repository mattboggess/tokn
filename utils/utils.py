from spacy.tokens import DocBin
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage


def write_spacy_docs(docs, filepath):
    """ Writes serialized spacy docs to file.  
    
    Parameters
    ----------
    docs: list of spacy.tokens.doc.Doc
        List of spacy Docs to write to file 
    filepath: str
        File path to serialized spacy docs
    """
    doc_bin = DocBin()
    for doc in docs:
        doc_bin.add(doc)
        
    with open(filepath, "wb") as f:
        f.write(doc_bin.to_bytes())
    
def read_spacy_docs(filepath, nlp=None):
    """ Reads serialized spacy docs from a file into memory.
    
    Parameters
    ----------
    filepath: str
        File path to serialized spacy docs
    
    Returns
    -------
    list of spacy.tokens.doc.Doc
        List of spacy Docs loaded from file
    """
    
    if nlp is None:
        snlp = stanfordnlp.Pipeline(lang="en")
        nlp = StanfordNLPLanguage(snlp)
        
    with open(filepath, "rb") as f:
        data = f.read()
        
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs

def tag_bioes(tags, match_index, term_length):
    """ Updates tags for a text using the BIOES tagging scheme.

    B = beginning of term phrase
    I = interior of term phrase
    O = non-term
    E = end of term phrase
    S = singleton term

    Parameters
    ----------
    tags: list of str
        List of current BIOES tags for given tokenized text that we will be updating
    match_index: int
        Index at which the term was matched in the text
    term_length: int
        Number of tokens that compose the term ('cell wall' -> 2)

    Returns
    -------
    list of str
        Updated list of BIOES tags for the given tokenized text

    Examples
    --------

    >>> tag_bioes(['O', 'O', 'O', 'O'], 1, 2)
    ['O', 'B', 'E', 'O']

    >>> tag_bioes(['O', 'O', 'O', 'O'], 0, 1)
    ['S', 'O', 'O', 'O']

    >>> tag_bioes(['O', 'O', 'O', 'O'], 1, 3)
    ['O', 'B', 'I', 'E']

    """

    if term_length == 1:
        tags[match_index] = "S"
    else:
        for i in range(term_length):
            if i == 0:
                tags[match_index + i] = "B"
            elif i == term_length - 1:
                tags[match_index + i] = "E"
            else:
                tags[match_index + i] = "I"
    return tags
