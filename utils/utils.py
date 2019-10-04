import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage


def tag_terms(text, terms):
    """ Identifies and tags any terms present in a given input text.

    Searches through the input text and finds all terms (single words and phrases) that are present
    in the list of provided terms. Returns a list of found terms with indices as well as a BIOES tagged
    version of the sentence denoting where the terms are in the sentences. Uses spacy functionality
    to tokenize and lemmatize for matching text.

    Gives precedence to longer terms first so that terms that are part of a larger term phrase
    are ignored (i.e. match 'cell wall', not 'cell' within the phrase cell wall)

    Parameters
    ----------
    text: str | spacy.tokens.doc.Doc
        Input text that will be preprocessed using spacy and searched for terms
    terms: list of str | list of spacy.tokens.doc.Doc
        List of input terms that will be preprocessed using spacy. It is recommended that one
        preprocess by spacy ahead of time if you will be checking the same terms across many input texts.

    Returns
    -------
    tuple
        First element: list of found terms each with list of indices where matches were found
        Second element: tokenized text as list of tokens
        Third element list of BIOES tags for the tokenized text

    Examples
    --------

    >>> tag_terms('The cell wall provides structure.', ['cell wall', 'cell'])
    ([{'cell wall': [1]}], ['The', 'cell', 'wall', 'provides', 'structure', '.'], ['O', 'B', 'E', 'O', 'O', 'O'])

    """

    # preprocess with spacy if needed
    snlp = stanfordnlp.Pipeline(lang="en")
    nlp = StanfordNLPLanguage(snlp)
    if type(terms[0]) != spacy.tokens.doc.Doc:
        terms = [nlp(term) for term in terms]
    if type(text) != spacy.tokens.doc.Doc:
        text = nlp(text)

    normalized_text = [token.lemma_ for token in text]
    tokenized_text = [token.text for token in text]

    # storage variables
    tagged_text = ['O'] * len(text)
    found_terms = {}

    # iterate through terms from longest to shortest
    terms = sorted(terms, key=len)[::-1]
    for term in terms:
        term_indices = []
        normalized_term = [token.lemma_ for token in term]

        for ix in range(len(text) - len(term)):

            if normalized_text[ix:ix + len(term)] == normalized_term:
                # only add term if not part of larger term
                if tagged_text[ix:ix + len(term)] == ["O"] * len(term):
                    term_indices.append(ix)
                    tagged_text = tag_bioes(tagged_text, ix, len(term))

        if len(term_indices):
            found_terms[term.text] = term_indices

    return (found_terms, tokenized_text, tagged_text)

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

