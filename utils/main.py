import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import time
import math


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import preprocessing


def url_to_html(url):
    """Scrapes the html content from a web page. Takes a URL string as input and returns an html object. """
    
    # Get the html content
    res = requests.get(url, headers={"User-Agent": "XY"})
    html = res.text
    parser_content = BeautifulSoup(html, 'html.parser')
    return parser_content


def get_parent_category_i(cat):
    return math.floor(int(cat) / 10) * 10

def size_mb(docs):
    """Calculate the size in megabytes (MB) of a list of documents.

    Args:
        docs (list of str): A list of documents as strings.

    Returns:
        float: The size of the documents in megabytes (MB).
    """
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

def tfidf_it(data, verbose=False):
    s = time()

    tfidf_vec = TfidfVectorizer(min_df=3, # If a token appears fewer times than this, across all documents, it will be ignored
                                 # tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                                 tokenizer=preprocessing.tokenize_data, # we use the custom tokenizer
                                 stop_words='english')#stopwords.words('english')) # stopwords are removed

    tfidf_text = tfidf_vec.fit_transform(data)


    e = time()
    
    if verbose:
        print("Elapsed time during the whole program in seconds:",
                                             e-s) 
    return tfidf_vec, tfidf_text

"""
load_vectorize_df(if df, don't load, use it.)
"""
def load_dataset(file_path, 
                 verbose=False, 
                 balanced=False, 
                 chosen_categories = [220, 230, 240],
                 exclude=[],
                 specific_cat=None,
                 target_label='parent_ocms',
                 exact=False,
                 sample=None):
    """Load and vectorize the 20 newsgroups dataset."""
    # CHECK OUT: 310, 340, 400, 520, 570, 580, 870] activities, building structures, machines, recreation, interpersonal relations, marriage, education.
    # chosen_categories = [220, 230, 240]
    df = pd.read_csv(file_path, encoding='utf-8')
    
    df['ocms_list'] = df['ocms'].str.split()
    df['ocms_list'] = df['ocms_list'].apply(lambda x: [int(i) for i in x])
    
    if balanced:
        # same samples
        minimum = 1442
        data = []
        for cat in chosen_categories:
            data.append(df[df[target_label] == cat][:minimum])
        data = pd.concat(data)
    elif not specific_cat is None:
        # data = df[ df['ocms_list'].map(lambda x: np.all([get_parent_category_i(cat) == specific_cat for cat in x]) )]
        
        # if not exact:
            # data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and np.all([get_parent_category_i(cat) == get_parent_category_i(specific_cat) for cat in x])) 
            #                                           or (len(x) == 1 and x[0] == specific_cat) )]
            data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and len(x) == 1 and get_parent_category_i(x[0]) == get_parent_category_i(specific_cat) ) 
                                                      or (len(x) == 1 and x[0] == specific_cat) )]
        # else:
        #     data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] == specific_cat )]
            
    else:
        data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] in chosen_categories )]
    

    if sample:
        data = data[:sample]

        
        
    t0 = time()
    
 
    
    # Split data into training and testing sets
    test_data = data[target_label] 
    
    if specific_cat and exact:
        test_data = data['ocms_list'].apply(lambda x: 1 if len(x) == 1 and specific_cat in x else 0)
    
    
    
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, 
        max_df=0.5, 
        min_df=5, 
        tokenizer=lambda doc: preprocessing.tokenize_data(doc, exclude),
        stop_words="english"
    )
    
    tfidf_text = vectorizer.fit_transform(data['textrecord'])
    
    
    X_train, X_test, y_train, y_test = train_test_split(tfidf_text, test_data, test_size=0.2)
    
    duration_train = time() - t0
    
    # order of labels in `target_names` can be different from `categories`
    # target_names = data['parent_label_name'].unique() if target_label == 'parent_ocms' else data['label_name'].unique()
    target_names = data['ocms'].unique()# if target_label == 'parent_ocms' else data['label_name'].unique()
    if exact:
        target_names = [str(specific_cat), str(get_parent_category_i(specific_cat))]
    
    # Extracting features from the test data using the same vectorizer
    # t0 = time()
    # X_test = vectorizer.transform(data_test.data)
    # duration_test = time() - t0

    feature_names = vectorizer.get_feature_names_out()

    if verbose:

        # compute size of loaded data
        data_train_size_mb = size_mb(data['textrecord'])
        # data_test_size_mb = size_mb(data_test.data)

        print(
            f"{len(data['textrecord'])} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        # print(f"{len(data_test.data)} documents - {data_test_size_mb:.2f}MB (test set)")
        print(f"{len(target_names)} categories")
        print(
            f"vectorize training done in {duration_train:.3f}s "
            f"at {data_train_size_mb / duration_train:.3f}MB/s"
        )
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        # print(
        #     f"vectorize testing done in {duration_test:.3f}s "
        #     f"at {data_test_size_mb / duration_test:.3f}MB/s"
        # )
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return data, X_train, X_test, y_train, y_test, feature_names, target_names