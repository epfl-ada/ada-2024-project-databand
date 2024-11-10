import os
import pandas as pd

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