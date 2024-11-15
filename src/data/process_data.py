import json
import pandas as pd
from collections import Counter

TMDB_string_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']


def clean_string_to_list(df, string_columns):
    for col in string_columns:
        df[col] = df[col].fillna('')
        df[col] = df[col].str.split(", ")
        df[col] = df[col].apply(lambda x: [] if x == [''] else x)  # Make empty string list to empty list
    return df

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

def combine_dataframes(df_movies, df_plots, df_tmdb, common_columns, cutoffyear, how_merge):
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
    df_combined = pd.merge(df_cmu, df_tmdb, on=['clean_title', 'release_year'], suffixes=("_cmu", "_tmdb"), how=how_merge)
    for column in common_columns:
        # create a column with combined values
        df_combined[column] = df_combined[column + "_cmu"] + df_combined[column + "_tmdb"]

        # remove the other separate columns from the dataframes
        df_combined.drop(column + "_cmu", axis=1, inplace=True)
        df_combined.drop(column + "_tmdb", axis=1, inplace=True)

    # clean column names
    df_combined.drop('title_tmdb', axis=1, inplace=True)
    colnames = [str(x).replace('_tmdb', '') for x in df_combined.columns.tolist()]
    colnames = [str(x).replace('_cmu', '') for x in colnames]
    df_combined.columns = colnames
    df_combined['overview'] = df_combined['overview'].fillna(df_combined['summary'])

    ''' # we complement our dataset with movies from cutoff onwards with TMDB dataset
    df_tmdb_post2016 = df_tmdb[df_tmdb['release_year'] >= cutoffyear].copy()
    df_tmdb_post2016.loc[:, 'summary'] = df_tmdb_post2016['overview']

    df_combined = pd.concat([df_combined, df_tmdb_post2016], axis=0)'''
    df_combined.drop('clean_title', axis=1, inplace=True)

    return df_combined

def get_dvd_era(release_year, start_year, end_year):
    if release_year < start_year:
        return 'pre'
    elif release_year < end_year:
        return 'during'
    else:
        return 'post'

def annotate_dvd_era(df):
    df['dvd_era'] = df['release_year'].apply(get_dvd_era, args=(1997,2013))
    return df

def remove_genres(df, genres):
    for genre in genres:
        df = df[df['genres'].apply(lambda x: genre not in x)]
    return df

def create_cmu_tmdb_dataset(cmu_movies_path, plots_path, tmdb_path, how_merge):
    df_movies = pd.read_csv(cmu_movies_path)
    df_plots = pd.read_csv(plots_path)
    df_tmdb = pd.read_csv(tmdb_path)
    df_tmdb = clean_string_to_list(df_tmdb, TMDB_string_columns)
    common_columns = list(set(df_movies.columns.tolist()) & set(df_tmdb.columns.tolist()))
    common_columns.remove('release_year')
    common_columns.remove('title')
    df_combined = combine_dataframes(df_movies=df_movies, df_plots=df_plots, df_tmdb=df_tmdb,
                                     common_columns=common_columns, cutoffyear=2012, how_merge=how_merge)
    df_combined = annotate_dvd_era(df_combined)
    df_combined['overview'] = df_combined['summary'].copy()
    df_combined.fillna({'overview': ''}, inplace=True)
    return df_combined

def create_tmdb_dataset(tmdb_path):
    df_tmdb = pd.read_csv(tmdb_path)
    df_tmdb = clean_string_to_list(df_tmdb, TMDB_string_columns)
    df_tmdb = df_tmdb[(df_tmdb['runtime'] == 0) | (df_tmdb['runtime'] > 45)]
    df_tmdb = df_tmdb[df_tmdb['genres'].apply(lambda x: 'Documentary' not in x)]
    df_tmdb.fillna({'overview': ''}, inplace=True)
    return annotate_dvd_era(df_tmdb)
