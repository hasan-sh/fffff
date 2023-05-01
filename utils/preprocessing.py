import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def unneeded_tokens(tokens):
    unneeded = ['table', 'graphic']
    return [token for token in tokens if not token in unneeded]

def insufficient_info(text : str):
    text = text.strip()
    if text.startswith('~') and text.endswith('~'):
        return True
    return

def remove_punct(token):
    return re.sub(r'[^\w\s]', '', token)
    
# turn a doc into clean tokens
def tokenize_data(doc, exclude=[]):
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
    tokens = [word for word in tokens if len(word) > 1]
    
    return tokens