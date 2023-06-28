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

seed = 27 # ensuring reproducibility

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



def upsample(df, data):
    """
    Upsample the minority class in a dataset based on the 'ocms' column.

    Args:
        df (DataFrame): The original dataset.
        data (DataFrame): The dataset with class imbalance.

    Returns:
        DataFrame: The upsampled dataset.

    """
    if len(data['ocms'].value_counts()) > 2:
        # Upsample for multiple classes
        values = list(data['ocms'].value_counts().sort_values().items())
        _, maximum = values[-1]
        
        # Iterate over each class except the majority class
        for ocm, minimum in values[:-1]:
            ocm = int(ocm)
            
            # Filter the data for the current class and conditions
            data1 = df[df['ocms_list'].map(lambda x: len(x) != 1 and ocm in x and 
                                            np.all([get_parent_category_i(ocm) != get_parent_category_i(cat) for cat in x if cat != ocm]))]
            
            to_draw = min(maximum - minimum, len(data1))
            data1 = data1.sample(to_draw, random_state=seed)
            
            # Update the 'ocms' and 'ocms_list' columns
            data1['ocms'] = str(ocm)
            data1['ocms_list'] = data1['ocms'].apply(lambda x: [int(x)])
            
            # Concatenate the upsampled data with the original data
            data = pd.concat([data, data1])
    else:
        # Upsample for binary class
        ocm = int(data['ocms'].value_counts().idxmin())
        minimum = data['ocms'].value_counts().min()
        maximum = data['ocms'].value_counts().max()
        
        # Filter the data for the current class and conditions
        data1 = df[df['ocms_list'].map(lambda x: len(x) != 1 and ocm in x and 
                                        np.all([get_parent_category_i(ocm) != get_parent_category_i(cat) for cat in x if cat != ocm]))]
        
        to_draw = min(maximum - minimum, len(data1))
        data1 = data1.sample(to_draw, random_state=seed)
        
        # Update the 'ocms' and 'ocms_list' columns
        data1['ocms'] = str(ocm)
        data1['ocms_list'] = data1['ocms'].apply(lambda x: [int(x)])
        
        # Concatenate the upsampled data with the original data
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
                 sample=None,
                **kwargs):
    """
    Load and vectorize the eHRAF dataset.

    Args:
        file_path (str): The path to the dataset file.
        verbose (bool, optional): Whether to display verbose output. Defaults to False.
        balanced (bool, optional): Whether to balance the dataset. Defaults to True.
        chosen_categories (list, optional): List of chosen categories. Defaults to [220, 230, 240].
        cultures (list, optional): List of cultures to include. Defaults to None.
        exclude (list, optional): List of categories to exclude. Defaults to [].
        specific_cat (int, optional): Specific category to include. Defaults to None.
        target_label (str, optional): Target label column name. Defaults to 'parent_ocms'.
        exact (bool, optional): Whether to include exact matches only. Defaults to False.
        stopwords (bool, optional): Whether to use stopwords. Defaults to True.
        sample (int, optional): Number of samples to include. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the loaded data and the train-test split.

    """
                    
    df = pd.read_csv(file_path, encoding='utf-8')
    
    df['ocms_list'] = df['ocms'].str.split()
    df['ocms_list'] = df['ocms_list'].apply(lambda x: [int(i) for i in x])
    
    if not specific_cat is None:
        """
            -> if an instance doesn't have that specific cat, but its parent cat is the same as the parent cat of the given specific cat, it's chosen. 
            OR
            -> if an instance has only that specific cat, it's chosen.
        """
        data = df[ df['ocms_list'].map(lambda x: (specific_cat not in x and len(x) == 1 and get_parent_category_i(x[0]) == get_parent_category_i(specific_cat) ) 
                                                  or (len(x) == 1 and x[0] == specific_cat) )]
        if exact:
            
            data['ocms_list'] = data['ocms_list'].apply(lambda x: [specific_cat] if len(x) == 1 and specific_cat in x else [get_parent_category_i(specific_cat)])
            data['ocms'] = data['ocms_list'].apply(lambda x: str(x[0]))
            
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
        
    X_train, X_test, y_train, y_test = train_test_split(data['textrecord'], test_data, test_size=0.3, random_state=seed)

    duration_train = time() - t0
    
    
    target_names = data['ocms'].unique()# if target_label == 'parent_ocms' else data['label_name'].unique()
    if specific_cat and exact:
        target_names = [str(specific_cat), str(get_parent_category_i(specific_cat))]
    
    if verbose:
        # compute size of loaded data
        data_train_size_mb = size_mb(data['textrecord'])

        print(
            f"{len(data['textrecord'])} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        print(f"{len(target_names)} categories")
        
    return data, X_train, X_test, y_train, y_test, target_names



def filter_other_ocms(df, cat):
    """
    Filters out all other OCMs except the given cat.

    Args: 
        df (DataFrame): a dataframe
        cat (int): an OCM code to keep.

    Returns:
        df (DataFrame): A DataFrame containing the rows having one OCM only.
    """
    parent_cat = get_parent_category_i(cat)
    
    df['ocms'] = df['ocms_list'].apply(lambda x: str(list(filter(lambda i: parent_cat == get_parent_category_i(i), x))[0]))

    return df
