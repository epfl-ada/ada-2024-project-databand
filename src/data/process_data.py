import json
import pandas as pd

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
