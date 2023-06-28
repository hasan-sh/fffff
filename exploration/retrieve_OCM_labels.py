#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from utils import url_to_html
import json


if __name__ == "__main__":

    
    # Note: This may not work if the content of that page changes.
    html = url_to_html('https://hraf.yale.edu/resources/reference/outline-of-cultural-materials/#interactive-list-of-ocm-subjects-with-descriptions')
    
    
    # # Retrieve OCM Labels
    
    id_to_label = {}
    
    for div in html.select('.collapseomatic'):
        id, *text = div.text.split()
        id_to_label[id] = ' '.join(text).lower()
    
    
    # In[39]:
    
    
    # Open a file for writing
    with open("../data/id_to_label.json", "w") as outfile:
        # Write the dictionary to the file in JSON format
        json.dump(id_to_label, outfile)
    
    print('Mapped and saved IDs to OCM code labels.')
    
    
    # Per Category
    categories = []
    for i, div in enumerate(html.select('.collapseomatic')):
        if i == 0:
            continue
        id, *text = div.text.split()
        if 'noarrow' in div.attrs['class']:
            parent_cat = list(categories[-1].keys())[0]
            categories[-1][parent_cat].append((id, ' '.join(text).lower()))
        else:
            categories.append({' '.join(text).lower(): [] })
            
    
    # Open a file for writing
    with open("../data/labels_per_category.json", "w") as outfile:
        # Write the dictionary to the file in JSON format
        json.dump(categories, outfile)
    
    print('Mapped and saved subcateogires to parent categories.')
    
    print('Done!')
