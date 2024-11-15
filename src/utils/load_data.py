import os
import pandas as pd
import pyreadr

# GLOBAL VARS
MOVIE_HEADERS = ["Wikipedia movie ID", "Freebase ID", "Movie name", "Release date","Box office revenue",
              "Runtime","Languages" ,"Countries", "Genres"]
CHARACTER_HEADERS = ["Wikipedia movie ID", "Freebase movie ID", "Movie release date", "Character name",
                    "Actor DoB", "Actor gender", "Actor height (m)", "Actor ethnicity (Freebase ID)",
                     "Actor name", "Actor age at movie release", "Freebase character/actor map ID",
                     "Freebase character ID", "Freebase actor ID"]
PLOT_HEADERS = ["Wikipedia movie ID", "Summary"]

ROOT_DIR = './'
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def load_raw_data(filename, sep=',', headers = []):
    '''
    (str, str) -> pd.DataFrame
    Loads raw data, with fields separated by sep, into a dataframe
    '''
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename, sep=sep)
    if headers != []:
        df.columns = headers
    return df

def save_csv_data(df, filename):
    """Saves cleaned dataframe to a CSV file"""
    df.to_csv(filename, index=False)


def load_dvd_sales():
    result = pyreadr.read_r('data/raw/movies.RData')
    df_dvd_releases = next(iter(result.values()))
    df_dvd_releases['dvd_release_date'] = pd.to_datetime(
        df_dvd_releases[['dvd_rel_year', 'dvd_rel_month', 'dvd_rel_day']]
        .astype('Int64')
        .astype(str)
        .agg('-'.join, axis=1),
        errors='coerce'
    )
    df_dvd_releases = df_dvd_releases.dropna(subset=['dvd_release_date'])
    path = os.path.join('data', 'processed', 'dvd_releases.csv')
    df_dvd_releases[['dvd_release_date']].to_csv(path, sep=',',
                                                 index=False,
                                                 header=True)

# if __name__ == "__main__":
#     # Define file paths
#     files = ["movie.metadata.tsv", "character.metadata.tsv", "plot_summaries.txt"]
#     csv_files = ["movies.csv", "characters.csv", "plot_summaries.csv"]


#     for file, headers, csv_file in zip(files, [MOVIE_HEADERS, CHARACTER_HEADERS, PLOT_HEADERS], csv_files):
#         raw_data_path = os.path.join(DATA_DIR, 'raw', file)
#         processed_data_path = os.path.join(DATA_DIR, 'processed',csv_file)

#         # Load raw data
#         raw_data = load_raw_data(raw_data_path, '\t', headers)

#         # Save the csv data
#         save_csv_data(raw_data, processed_data_path)

#         print(f"Cleaned data saved to {processed_data_path}")
