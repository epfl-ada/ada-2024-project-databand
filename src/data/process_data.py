import json
import pandas as pd
from collections import Counter

def extract_tuples_values(json_string):
    """
    Given a json string, return the list of values in the dict
    >>> extract_tuples_values('{"/m/02h40lc": "English Language"}')
    ['English Language']
    """
    # convert json string to dict
    dictionary = json.loads(json_string)
    # extract values
    values_list = list(dictionary.values())
    return values_list

def extract_tuples_freebase_ids(json_string):
    """
    Given a json string, return the list of keys, i.e. freebase ids, in the dict
    >>> extract_tuples_values('{"/m/02h40lc": "English Language"}')
    ['/m/02h40lc']
    """
    # convert json string to dict
    dictionary = json.loads(json_string)
    # extract values
    keys_list = list(dictionary.keys())
    return keys_list

def get_sorted_counts(list_of_values, cut_off = None, bigger = True):
    """
    From list of values, return the values and their counts in decreasing order
    If a cut off is specified, only return values bigger or smaller (depending on bigger parameter) than the cutoff
    By default, there is no cutoff, and if it is specified, the default is to return values bigger than it
    """
    if cut_off is not None and bigger:
        all_freqs = {k: v for k, v in Counter(list_of_values).items() if v > cut_off}
    elif cut_off is not None:
        all_freqs = {k: v for k, v in Counter(list_of_values).items() if v <= cut_off}
    else:
        all_freqs = Counter(list_of_values)

    sorted_counts = sorted(all_freqs.items(), key = lambda x: x[1], reverse = True)
    values, counts = zip(*sorted_counts)

    return values, counts
