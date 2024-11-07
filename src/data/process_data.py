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

def extract_cols_values(df, column_names):
    for col in column_names:
        df[col] = df[col].apply(extract_tuples_values)

def get_sorted_counts(list_of_values, cut_off = None, bigger = True):
    """
    From list of values, return the values and their counts in decreasing order
    If a cut-off is specified, only return values bigger or smaller (depending on bigger parameter) than the cutoff
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

def combine_dataframes(df_movies, df_plots, df_tmdb, common_columns):
    """
    For CMU, combine movie and plot summaries dataframes
    Given CMU and TMDB dataframes, perform an inner merge based on the movie title name and year of release
    Then combine "duplicate" columns from common_columns
    """
    # combine movie and plot dataframes based on wikiID, then remove that column
    df_cmu = pd.merge(df_movies, df_plots, on='wikipedia_movie_id')
    df_cmu.drop('wikipedia_movie_id', axis=1, inplace=True)

    # for CMU and TMDB movies merge based on movie title in lowercase and release date
    df_cmu['clean_title'] = df_cmu['title'].str.lower().str.strip()
    df_tmdb['clean_title'] = df_tmdb['title'].str.lower().str.strip()
    df_combined = pd.merge(df_cmu, df_tmdb, on=['clean_title', 'release_year'], suffixes=("_cmu", "_tmdb"))


    for column in common_columns:
        # create a column with combined values
        df_combined[column] = df_combined[column + "_cmu"] + df_combined[column + "_tmdb"]

        # remove the other separate columns from the dataframes
        df_combined.drop(column + "_cmu", axis=1, inplace=True)
        df_combined.drop(column + "_tmdb", axis=1, inplace=True)

    # clean column names
    colnames = [str(x).replace('_tmdb', '') for x in df_combined.columns.tolist()]
    colnames = [str(x).replace('_cmu', '') for x in colnames]
    df_combined.columns = colnames

    # we complement our dataset with movies from 2016 onwards with TMDB dataset
    df_tmdb_post2016 = df_tmdb[df_tmdb['release_year'] >= 2016]
    df_tmdb_post2016['summary'] = df_tmdb_post2016['overview']

    df_combined = pd.concat([df_combined, df_tmdb_post2016], axis=1)

    return df_combined