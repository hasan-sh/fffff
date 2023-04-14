import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import time


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



def size_mb(docs):
    """Calculate the size in megabytes (MB) of a list of documents.

    Args:
        docs (list of str): A list of documents as strings.

    Returns:
        float: The size of the documents in megabytes (MB).
    """
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6



def load_dataset(file_path, verbose=False, balanced=False, target_label='parent_ocms', sample=None):
    """Load and vectorize the 20 newsgroups dataset."""
    # CHECK OUT: 310, 340, 400, 520, 570, 580, 870] activities, building structures, machines, recreation, interpersonal relations, marriage, education.
    chosen_categories = [140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 360, 420, 430, 590, 620]#, 780, 820]
    df = pd.read_csv(file_path)
    
    if balanced:
        # same samples
        minimum = 1442
        data = []
        for cat in chosen_categories:
            data.append(df[df[target_label] == cat][:minimum])
        data = pd.concat(data)
    else:
        data = df[df[target_label].isin(chosen_categories)]
        
    if sample:
        data = data[:sample]

        
        
    t0 = time()
    
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, 
        max_df=0.5, 
        min_df=5, 
        tokenizer=preprocessing.tokenize_data,
        stop_words="english"
    )
    
    tfidf_text = vectorizer.fit_transform(data['textrecord'])
    
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_text, data[target_label], test_size=0.2)
    
    duration_train = time() - t0
    
    # order of labels in `target_names` can be different from `categories`
    target_names = data['parent_label_name'].unique() if target_label == 'parent_ocms' else data['label_name'].unique()



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