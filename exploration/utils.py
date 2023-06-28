import string
from collections import Counter
import math
import requests
from bs4 import BeautifulSoup

def replace_punctuation_with_space(text):
    """
    Replace punctuation in a text with spaces.

    Args:
        text (str): The text to process.

    Returns:
        str: The text with punctuation replaced by spaces.
    """
    # Create a translation table with punctuation replaced by spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Use the translation table to replace punctuation with spaces
    return text.translate(translator)


def get_ocm_counts(df, id_to_category=None, limit=626):
    """
    Get the counts of OCMs (Organized Crime Members) in a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the 'ocms' column.
        id_to_category (dict, optional): A dictionary mapping OCM IDs to categories. Defaults to None.
        limit (int, optional): The maximum OCM ID to consider. Defaults to 626.

    Returns:
        Counter: The counts of OCMs.
    """
    ocms_list = df['ocms'].str.split(expand=True).values.ravel()
    ocms_list = [id_to_category[int(i)] if id_to_category else int(i) for i in ocms_list if i and int(i) <= limit]
    ocms_counts = Counter(ocms_list)
    return ocms_counts


def make_id_category(ocms, label_to_id):
    """
    Create a dictionary mapping OCM IDs to categories.

    Args:
        ocms (list): List of OCMs.
        label_to_id (dict): A dictionary mapping OCM categories to IDs.

    Returns:
        dict: A dictionary mapping OCM IDs to categories.
    """
    id_to_category = {000: 000}
    for item in ocms:
        category, values = list(item.items())[0]
        cat_id = label_to_id[category]

        id_to_category[cat_id] = cat_id
        for x in values:
            id_to_category[int(x[0])] = cat_id
    return id_to_category

def get_parent_category_i(cat):
    """
    Get the parent category ID for a given category ID.

    Args:
        cat (int): The category ID.

    Returns:
        int: The parent category ID.
    """
    return math.floor(int(cat) / 10) * 10



def url_to_html(url):
    """
    Scrapes the HTML content from a web page.

    Args:
        url (str): The URL of the web page.

    Returns:
        BeautifulSoup: An object representing the parsed HTML content.
    """
    # Get the HTML content
    res = requests.get(url, headers={"User-Agent": "XY"})
#     res = requests.get(url + ".pdf", headers={"User-Agent": "XY"})

    html = res.text

    # Parse the HTML content
    parser_content = BeautifulSoup(html, 'html5lib')

    return parser_content

    

