import json
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from textblob import TextBlob

# Textbooks on Open Stax to Parse
# You can add more open stax textbooks to parse by adding a key-value pair with
# the outline url as a value. Make sure the parsing assumptions hold or edit to handle special
# cases as needed
TEXTBOOKS = {'astronomy': 'https://cnx.org/contents/LnN76Opl@20.1:_45u6IpQ@8/Introduction',
             'anatomy': 'https://cnx.org/contents/FPtK1zmh@15.1:zMTtFGyH@7/Introduction',
             'biology': 'https://cnx.org/contents/jVCgr5SL@15.3:IjCrkDE3@9/Introduction',
             'chemistry': 'https://cnx.org/contents/f8zJz5tx@6.1:DY-noYmh@6/Introduction',
             'physicsv1': 'https://cnx.org/contents/1Q9uMg_a@13.3:bG-_rWXy@9/Introduction',
             'physicsv2': 'https://cnx.org/contents/eg-XcBxE@16.1:DSrCtKWS@6/Introduction',
             'physicsv3': 'https://cnx.org/contents/rydUIGBQ@12.1:bq-wv5M8@6/Introduction',
             'microbiology': 'https://cnx.org/contents/5CvTdmJL@7.1:rFziotaH@5/Introduction'}


def get_page_html(driver, link, delay=2, max_tries=5):
    """ Retrieves html representation of link using selenium webdriver that can execute javascript.

    This will retrieve the html at the address represented by link
    using the provided selenium webdriver that can execute javascript. It will wait a
    number of seconds given by delay to ensure the web page loads since the textbook pages
    are javascript generated and take a few seconds to generate the html content. If it
    detects that the webpage does not load in this time it will re-attempt with an additional
    second added to delay for several tries.

    Parameters
    ----------
    driver: selenium webdriver
        Selenium webdriver object that will retrieve the page
    link: str
        Address of the page that one wants to get the html from
    delay: int, optional
        Number of seconds to initially delay the webdriver retrieval to ensure the page loads
    max_tries: int, optional
        The number of times to re-attempt the page loading with a delay increased by 1 second each
        time.

    Returns
    -------
    soup: BeautifulSoup object
        BeautifulSoup object representing parsed html found at link

    Raises
    ------
    RuntimeError
        If the page did not load with the provided number of tries and starting delay
    """

    html = 'This site requires Javascript.'
    num_tries = 0

    # repeatedly try fetching with a 1 sec longer delay each time for page load
    while 'This site requires Javascript.' in html and num_tries <= max_tries:
        num_tries += 1
        driver.get(link)

        # hack to ensure that the webdriver waits at least delay seconds for the code to load
        # essentially waits for an element with empty id to occur, which won't happen but on
        # timeout it just continues to execute the code with the page now loaded.
        # the page may be loaded in less than than delay seconds but would require a page-specific
        # id so we wait the full delay seconds to ensure each page fully loads
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, '')))
        except:
            pass

        html = driver.page_source
        delay += 1

    # raise error if we still couldn't fetch page in the maximum attempted delay
    if 'This site requires Javascript.' in html:
        raise RuntimeError('Could not load site %s with a max tried delay of %d seconds' % (link, delay))

    soup = BeautifulSoup(html, features='html.parser')
    return soup

def extract_ch_links(soup):
    """ Extracts all links to chapter text given a BeautifulSoup representation of outline page.

    Parameters
    ----------

    soup: BeautifulSoup object
        BeautifulSoup representation of the outline page html

    Returns
    -------

    ch_links: list of str
        Returns a list of links to chapter text pages in sorted order
    """

    # filter to chapter links by only including those with a chapter number in text
    ch_nums = soup.find_all('span', {'class': 'os-number'})
    ch_links = {ch_num.text : ch_num.parent.parent.get('href') for ch_num in ch_nums
                if ch_num.parent.parent.get('href')}

    # sort by chapter order
    link_keys = sorted([float(l) for l in ch_links.keys() if l[0].isdigit()])
    ch_links = [ch_links[str(lk)] for lk in link_keys] # sorted in chapter order
    ch_links = ['https://cnx.org' + ch_link for ch_link in ch_links]

    return ch_links

def extract_key_terms(key_term_soup, glossary, textbook):
    """ Extracts key terms, their acronym representation, and their glossary definition
    from a provided web page and updates a glossary dictionary.

    Parameters
    ----------

    key_term_soup: BeautifulSoup object
        BeautifulSoup representation of the web page containing the key terms

    glossary: dictionary [str -> dictionary]
        Dictionary mapping terms to a dictionary containing acronym representation
        and glossary definition

    textbook: string
        Name of textbook being processed to allow for special case handling

    Returns
    -------

    glossary: dictionary [str -> dictionary]
        Updated dictionary mapping terms to a dictionary containing acronym representation
        and glossary definition
    """

    if textbook == 'microbiology':
        term_entries = [t for t in key_term_soup.find_all('li')
                        if t.findChildren()[0].name == 'strong']
        terms = [t.findChildren()[0].text.strip() for t in term_entries]
        gloss_defs = [t.text.split(t.findChildren()[0].text)[-1].replace('"', '').strip() for t in term_entries]
    else:
        term_entries = [t for t in key_term_soup.find_all('dl') if t.get('id') and 'fs-id' in t.get('id')]
        terms = [t.findChildren()[0].text for t in term_entries]
        gloss_defs = [t.findChildren()[1].text.replace('"', '') for t in term_entries]

    for term, definition in zip(terms, gloss_defs):
        term, acronym = extract_acronym(term)
        if term in glossary.keys():
            glossary[term]['glossary_definition'].append(definition)
        else:
            glossary[term] = {'acronym': acronym,
                              'glossary_definition': [definition],
                              'text_definition': []}
    return glossary


def extract_acronym(term):
    """ Extract acronym representation from a textbook key term.

    Assumes acronyms/alternate representations are encoded in parentheses.
    This will replace whatever is to the left of the parentheses for the
    acronym replacement. For example:

        ribonucleic acid (RNA) synthase -> RNA synthase

    Parameters
    ----------

    term: string
        The key term to be processed

    Returns
    -------

    (clean_term, acronym): tuple
        Tuple containing the cleaned key term and its acronym representation.
        Returns an empty string for acronym if none is present in the term
    """
    term = re.split('\(|\)', term)
    if len(term) <= 2:
        # no parens and thus no acronym
        clean_term = term[0].strip()
        acronym = ''
    else:
        clean_term = term[0].strip() + ' ' + term[2].strip()
        acronym = term[1].strip() + ' ' + term[2].strip()

    clean_term = clean_term.strip()
    acronym = acronym.strip()
    return clean_term, acronym

def extract_term_defs(elems, glossary):
    """ Extract term definition sentence from a paragraph of chapter text.

    Searchs for elements in the input elems list that match the pattern of
    being a bolded key term in the text and extracts the sentence containing
    this term into the glossary.

    Parameters
    ----------

    elems: list of BeautifulSoup Tag objects
        List of html tag objects that together compose a full paragraph of text.

    glossary: dictionary [str -> dictionary]
        Dictionary mapping terms to a dictionary containing acronym representation
        and glossary definition and text definition

    Returns
    -------

    glossary: dictionary [str -> dictionary]
        Updated dictionary mapping terms to a dictionary containing acronym representation
        and glossary definition and text definition
    """

    for i, elem in enumerate(elems):
        if hasattr(elem, 'get') and elem.get('id'):
            if 'term' in elem.get('id') and not elem.get('class'):
                term = elem.text

                # build front of sentence (iterate until period)
                front = ''
                j = i - 1
                while j >= 0:
                    if hasattr(elems[j], 'text'):
                        front = elems[j].text + front
                    elif '.' in elems[j]:
                        front = elems[j].split('.')[-1] + front
                        break
                    else:
                        front = elems[j] + front
                    j -= 1

                # build end of sentence (iterate until period)
                end = ''
                j = i + 1
                while j < len(elems):
                    if hasattr(elems[j], 'text'):
                        end = end + elems[j].text
                    elif '.' in elems[j]:
                        end = end + elems[j].split('.')[0]
                        break
                    else:
                        end = end + elems[j]
                    j += 1

                term_def = front + term + end + '.'
                term_def = term_def.strip()

                # match term to corresponding term in glossary
                clean_term, _ = extract_acronym(term)
                clean_term = clean_term.strip()
                if clean_term not in glossary.keys():
                    if len(clean_term) > 1:
                        lower_term = clean_term[0].lower() + clean_term[1:]
                    else:
                        lower_term = clean_term.lower()

                    words = TextBlob(clean_term).words
                    words = [word.singularize() for word in words]
                    single_term = ' '.join(words)
                    words = TextBlob(lower_term).words
                    words = [word.singularize() for word in words]
                    single_lower_term = ' '.join(words)

                    # handle capitalized terms
                    if lower_term in glossary.keys():
                        clean_term = lower_term

                    # handle plurals
                    elif single_term in glossary.keys():
                        clean_term = single_term

                    # handle plurals + capitalization
                    elif single_lower_term in glossary.keys():
                        clean_term = single_lower_term
                    else:
                        print('WARNING: Skipping non-glossary bolded word: %s' % clean_term)
                        continue

                glossary[clean_term]['text_definition'].append(term_def)

    return glossary

def extract_ch_sentences(paragraph):
    """ Extract chapter sentences from a paragraph of text.

    Converts a pargraph of text into a list of sentences where sentences
    are split using nltk's sent_tokenize. Each sentence is then split into tokens
    using nltk's word_tokenize and then re-separated by a space.

    Parameters
    ----------

    paragraph: string
        String containing a paragraph of text from the textbook

    Returns
    -------

    sents: List of strings
        List of strings where each entry is a separate sentence where tokens in that sentence are
        separated by spaces
    """
    sents = sent_tokenize(paragraph)
    sents = [word_tokenize(sent) for sent in sents]

    # screen out too short and long sentences
    sents = [sent for sent in sents if len(sent) > 3]
    sents = [sent for sent in sents if len(sent) < 50]

    sents = [' '.join(sent) for sent in sents]
    return sents

def extract_chapter_text(ch_soup, glossary):
    """ Extract chapter sentences and term definition sentences from a chapter web page.

    Parameters
    ----------

    ch_soup: BeautifulSoup object
        BeautifulSoup representation of chapter section web page html

    Returns
    -------

    glossary: dictionary [str -> dictionary]
        Updated dictionary mapping terms to a dictionary containing acronym representation
        and glossary definition and text definition
    """

    ch_sents = []
    ch_para = [p for p in ch_soup.find_all('p', {'id': re.compile('fs-id.*')})]
    for p in ch_para:
        para_text = p.text
        elems = p.contents

        # update glossary with text definition sentences
        glossary = extract_term_defs(elems, glossary)

        # extract sentences from paragraphs
        ch_sents += extract_ch_sentences(para_text)

    return ch_sents, glossary


if __name__ == '__main__':

    # set up selenium web driver: headless chrome w/o loading images
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(executable_path='./chromedriver', chrome_options=chrome_options)

    for textbook, outline_link in TEXTBOOKS.items():
        print('Processing Textbook: %s' % textbook)

        # get html of textbook outline page
        outline_soup = get_page_html(driver, outline_link)

        # extract all links to pages containing key terms
        if textbook == 'microbiology':
            # microbiology book has a separate glossary section
            pattern = 'Glossary'
        else:
            # other textbooks have individual key terms sections for each chapter
            pattern = 'Key Terms'
        links = outline_soup.find_all('a')
        kt_links = set(['https://cnx.org' + l.get('href') for l in links if pattern in l.text])

        # extract glossary terms and glossary definitions from each of the glossary pages
        glossary = {}
        for i, kt_link in enumerate(kt_links):
            print('Processing Key Term Page %d of %d' % (i + 1, len(kt_links)))
            kt_soup = get_page_html(driver, kt_link)
            glossary = extract_key_terms(kt_soup, glossary, textbook)

        # extract all links to chapter sections
        ch_links = extract_ch_links(outline_soup)

        # extract chapter sentences and textbook term definition sentences from each chapter section page
        textbook_sents = []
        for i, ch_link in enumerate(ch_links):
            print('Processing Ch. Text Page %d of %d' % (i + 1, len(ch_links)))
            ch_soup = get_page_html(driver, ch_link)
            ch_sents, glossary = extract_chapter_text(ch_soup, glossary)
            textbook_sents += ch_sents

        # save data
        with open('data/%s_chapter_sentences.txt' % textbook, 'w', encoding='utf8') as fid:
            fid.write('\n'.join(textbook_sents))
        with open('data/%s_glossary.json' % textbook, 'w', encoding='utf8') as fid:
            json.dump(glossary, fid, indent=4, ensure_ascii=False)

    driver.quit()

