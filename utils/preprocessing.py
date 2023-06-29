import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
import spacy

nlp = spacy.load("en_core_web_sm")


stop_words = set(stopwords.words('english'))


POSSIBLY_NEEDED_STOPWORDS = """
he
she
i
his
himself
her
hers
herself
my
mine

you
they
we
your
yours
yourself
yourselves
their
theirs
them
themselves
our
ours

alone
""".split()

nlp.Defaults.stop_words -= set(POSSIBLY_NEEDED_STOPWORDS) # Keep needed (possibly) stopwords.


def unneeded_tokens(tokens):
    """
    Remove unneeded tokens from a list of tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of tokens without unneeded tokens.
    """
    unneeded = ['table', 'graphic', 'caption']
    return [token for token in tokens if not token in unneeded]


def insufficient_info(text: str):
    """
    Check if a text contains insufficient information.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text contains insufficient information, False otherwise.
    """
    text = text.strip()
    if text.startswith('~') and text.endswith('~'):
        return True
        
    if text.startswith('[') and text.endswith(']'):
        return True
    return False


def remove_punct(token):
    """
    Remove punctuation from a token.

    Args:
        token (str): The token to remove punctuation from.

    Returns:
        str: The token without punctuation.
    """
    return re.sub(r'[^\w\s]', '', token)


def tokenize_data(doc, **kwargs):
    """
    Tokenize a document and apply various filters to the tokens.

    Args:
        doc (str): The document to tokenize.
        exclude (list, optional): List of words to exclude from the tokens. Defaults to None.

    Returns:
        list: List of tokens.
    """
    if insufficient_info(doc):
        return []

    tokens = nlp(doc, disable=['parser', 'ner'])

    # Filter out punctuation, stopwords, and only keep numeric and alphabetical tokens.
    tokens = [token.lemma_ for token in tokens if 
              (token.is_alpha or (token.is_digit and len(token) > 3)) and # only years may be useful.
              not token.is_punct and 
              not token.is_stop
              # (not token.is_stop or (token.is_stop and not token.text in POSSIBLY_NEEDED_STOPWORDS))
             ] 
    
    # Filter out edge-case words.
    tokens = unneeded_tokens(tokens)
    
    if kwargs.get('exclude'):
        tokens = [word for word in tokens if not np.any([re.match(word, t) for t in kwargs.get('exclude')])]
    return tokens

# Turn a doc into clean tokens.
def tokenize_data_legacy(doc, **kwargs):
    if insufficient_info(doc):
        return []
    # print(doc)
    # doc = re.sub(r'[^\w\s]', '', doc)
    # print(doc)
    # split into tokens by white space
    tokens = doc.split()
    
    # filter out edge-case words.
    # tokens = unneeded_tokens(tokens)
    
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) if not remove_punct(w).isnumeric() else w 
              for w in tokens]

    # lemmatize words.
    # tokens = [get_lemma(w) for w in tokens]

    if kwargs.get('stopwords'):
        # filter out stop words
        # tokens = [w for w in tokens if not remove_punct(w).lower() in stop_words]
        tokens = [w for w in tokens if not w in stop_words]
    
    # remove remaining tokens that are not alphabetic or numeric
    tokens = [word  for word in tokens if remove_punct(word).isalpha()
              # or word.replace('.', '',).replace(',', '').isnumeric()]
              or word.isnumeric()]
    # remove punctuation of actual words; like apostrophies'
    tokens = [remove_punct(word) if remove_punct(word).isalpha() else word  for word in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word.strip()) > 1]
    
    if kwargs.get('exclude'):
        tokens = [word for word in tokens if not np.any([re.match(word, t) for t in kwargs.get('exclude')])]
    return tokens

