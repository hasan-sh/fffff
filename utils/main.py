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

"""
if two specific cats or chosen cats are given and some have less samples, we could upsample from the ones with fewer instances:
    
"""


def upsample(df, data):
    if len(data['ocms'].value_counts()) > 2:
        values = list(data['ocms'].value_counts().sort_values().items())
        _, maximum = values[-1]
        for ocm, minimum in values[:-1]:
            ocm = int(ocm)
            data1 = df[ df['ocms_list'].map(lambda x: len(x) != 1 and ocm in x and 
                                    np.all([get_parent_category_i(ocm) != get_parent_category_i(cat) for cat in x if cat != ocm]) )]
            
            to_draw = min(maximum - minimum, len(data1))
            data1 = data1.sample(to_draw)
            data1['ocms'] = str(ocm)
            data1['ocms_list'] = data1['ocms'].apply(lambda x: [int(x)])
            data = pd.concat([data, data1])
    else:
        ocm = int(data['ocms'].value_counts().idxmin())
        minimum = data['ocms'].value_counts().min()
        maximum = data['ocms'].value_counts().max()
        
        data1 = df[ df['ocms_list'].map(lambda x: len(x) != 1 and ocm in x and 
                                    np.all([get_parent_category_i(ocm) != get_parent_category_i(cat) for cat in x if cat != ocm]) )]
        
        to_draw = min(maximum - minimum, len(data1))
        data1 = data1.sample(to_draw)
        data1['ocms'] = str(ocm)
        data1['ocms_list'] = data1['ocms'].apply(lambda x: [int(x)])
        data = pd.concat([data, data1])
    return data
    
                     
def load_dataset(file_path, 
                 verbose=False, 
                 balanced=True, 
                 chosen_categories = [220, 230, 240],
                 cultures=None,
                 exclude=[],
                 specific_cat=None,
                 target_label='parent_ocms',
                 exact=False,
                 stopwords=True,
                 sample=None):
    """Load and vectorize the eHRAF dataset."""
    # CHECK OUT: 310, 340, 400, 520, 570, 580, 870] activities, building structures, machines, recreation, interpersonal relations, marriage, education.
    # chosen_categories = [220, 230, 240]
    df = pd.read_csv(file_path, encoding='utf-8')
    
    df['ocms_list'] = df['ocms'].str.split()
    df['ocms_list'] = df['ocms_list'].apply(lambda x: [int(i) for i in x])
    
    if not specific_cat is None:
        # data = df[ df['ocms_list'].map(lambda x: np.all([get_parent_category_i(cat) == specific_cat for cat in x]) )]
        
        # if not exact:
            # data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and np.all([get_parent_category_i(cat) == get_parent_category_i(specific_cat) for cat in x])) 
            #                                           or (len(x) == 1 and x[0] == specific_cat) )]
            """
                -> if an instance doesn't have that specific cat, but its parent cat is the same as the parent cat of the given specific cat, it's chosen. 
                OR
                -> if an instance has only that specific cat, it's chosen.
            """
            data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and len(x) == 1 and get_parent_category_i(x[0]) == get_parent_category_i(specific_cat) ) 
                                                      or (len(x) == 1 and x[0] == specific_cat) )]
        # else:
        #     data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] == specific_cat )]
            
    else:
        """
            -> if an instance has one OCM code that's in the chosen cats, it's chosen.
        """
        data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] in chosen_categories )]
    

    if balanced:
        data = upsample(df, data)
        
    if sample:
        data = data[:sample]

    if cultures:
        data = data[ data['culture'].map(lambda culture: np.any([culture == cul for cul in cultures])) ]
        
        
        
    t0 = time()
    
 
    
    # Split data into training and testing sets
    test_data = data[target_label] 
    
    if specific_cat and exact:
        # test_data = data['ocms_list'].apply(lambda x: 1 if len(x) == 1 and specific_cat in x else 0)
        data['ocms_list'] = data['ocms_list'].apply(lambda x: specific_cat if len(x) == 1 and specific_cat in x else get_parent_category_i(specific_cat))
        data['ocms'] = data['ocms_list'].apply(lambda x: str(specific_cat) if len(x) == 1 and specific_cat in x else str(get_parent_category_i(specific_cat)))
        test_data = data['ocms_list'].apply(lambda x: str(specific_cat) if len(x) == 1 and specific_cat in x else str(get_parent_category_i(specific_cat)))
        
    print(data['textrecord'].shape, test_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(data['textrecord'], test_data, test_size=0.3)
        
    # vectorizer = TfidfVectorizer(
    #     sublinear_tf=True, 
    #     max_df=0.5, 
    #     min_df=5, 
    #     tokenizer=lambda doc: preprocessing.tokenize_data(doc, exclude=exclude, stopwords=stopwords),
    #     # stop_words="english"
    # )
    
    # X_test_tra = vectorizer.fit_transform(data['textrecord'])

    duration_train = time() - t0
    
    # order of labels in `target_names` can be different from `categories`
    # target_names = data['parent_label_name'].unique() if target_label == 'parent_ocms' else data['label_name'].unique()
    target_names = data['ocms'].unique()# if target_label == 'parent_ocms' else data['label_name'].unique()
    if specific_cat and exact:
        target_names = [str(specific_cat), str(get_parent_category_i(specific_cat))]
    
    # Extracting features from the test data using the same vectorizer
    # t0 = time()
    # X_test = vectorizer.transform(data_test.data)
    # duration_test = time() - t0

    # feature_names = vectorizer.get_feature_names_out()

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
        # print(
        #     f"vectorize training done in {duration_train:.3f}s "
        #     f"at {data_train_size_mb / duration_train:.3f}MB/s"
        # )
        # print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        # # print(
        # #     f"vectorize testing done in {duration_test:.3f}s "
        # #     f"at {data_test_size_mb / duration_test:.3f}MB/s"
        # # )
        # print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return data, X_train, X_test, y_train, y_test, target_names


def filter_other_ocms(df, cat):
    parent_cat = get_parent_category_i(cat)
    
    df['ocms'] = df['ocms_list'].apply(lambda x: str(list(filter(lambda i: parent_cat == get_parent_category_i(i), x))[0]))

    return df


def load_dataset_genearal(file_path, 
                 verbose=False, 
                 balanced=True, 
                 chosen_categories = [220, 230, 240],
                 exclude=[],
                 specific_cat=None,
                 target_label='parent_ocms',
                 exact=False,
                 stopwords=True,
                 sample=None):
    """Load and vectorize the 20 newsgroups dataset."""
    # CHECK OUT: 310, 340, 400, 520, 570, 580, 870] activities, building structures, machines, recreation, interpersonal relations, marriage, education.
    # chosen_categories = [220, 230, 240]
    df = pd.read_csv(file_path, encoding='utf-8')
    
    df['ocms_list'] = df['ocms'].str.split()
    df['ocms_list'] = df['ocms_list'].apply(lambda x: [int(i) for i in x])
    
    if not specific_cat is None:
        # data = df[ df['ocms_list'].map(lambda x: np.all([get_parent_category_i(cat) == specific_cat for cat in x]) )]
        
        # if not exact:
            # data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and np.all([get_parent_category_i(cat) == get_parent_category_i(specific_cat) for cat in x])) 
            #                                           or (len(x) == 1 and x[0] == specific_cat) )]
            data = df[ df['ocms_list'].map(lambda x: 
                                           (specific_cat not in x and 
                                            len([cat for cat in x if get_parent_category_i(cat) == get_parent_category_i(specific_cat)]) == 1 ) 
                                           
                                           or (specific_cat in x and 
                                                     not np.any([specific_cat != cat and 
                                                                 get_parent_category_i(cat) == get_parent_category_i(specific_cat) for cat in x]) )
                                          )]
            
            data = filter_other_ocms(data, specific_cat)
            
            # data['ocms'] = data['ocms'].apply(lambda x: )
        # else:
        #     data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] == specific_cat )]
            
    else:
        data = df[ df['ocms_list'].map(lambda x: len(x) == 1 and x[0] in chosen_categories )]
    

    if balanced:
        data = upsample(df, data)
        
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
        tokenizer=lambda doc: preprocessing.tokenize_data(doc, exclude=exclude, stopwords=stopwords),
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