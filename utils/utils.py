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
    
def read_spacy_docs(filepath):
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
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    with open(filepath, "rb") as f:
        data = f.read()
        
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs
