import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re
sys.path.append(str(Path().resolve()  / 'src' / 'utils'))
from load_data import load_raw_data, save_csv_data

# ROOT_DIR = './' 
# DATA_DIR = os.path.join(ROOT_DIR, 'data')

# with open(os.path.join(DATA_DIR, 'genres_to_clean'), 'r') as file:
#     lines = file.readlines()

# # Remove newline characters if they exist at the end of each line
# genres_to_split = [line.strip() for line in lines]



# character_headers = ["Wikipedia movie ID", "Freebase movie ID", "Movie release date", "Character name",
#                              "Actor DoB", "Actor gender", "Actor height (m)", "Actor ethnicity (Freebase ID)",
#                              "Actor name", "Actor age at movie release", "Freebase character/actor map ID",
#                              "Freebase character ID", "Freebase actor ID"]
plot_headers = ["wikipedia_movie_id", "summary"]

class DataCleaner:
    def __init__(self, required_columns, string_columns, numeric_columns):
        self.required_columns = required_columns
        self.string_columns = string_columns
        self.numeric_columns = numeric_columns
    
    def clean_status(self, df):
        """The movies must be released"""
        return df[df['status'] == 'Released']
    
    def clean_release_date(self, df):
        """Check if dates are in YYYY-MM-DD format"""
        
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_date'].dropna(inplace=True)
        df = df[df['release_date']>='1976-01-01'] # We want to keep movies from 1976 onwards, after the appearance of the VHS
        df = df[df['release_date'] <= '2024-01-01']  # We want to keep movies until 2024
        return df
    
    def clean_release_year(self, df):
        if 'release_year' not in df.columns:
            df['release_year'] = df['release_date'].apply(lambda x: str(x)[:4] if pd.notnull(x) else None)
        
        df['release_year'] = df['release_year'].astype(str).str[:4]
        df = df[df['release_year']>='1976']
        df = df[df['release_year'] <= '2024']
        return df

    
    def clean_numeric_columns(self, df):
        """Clean numeric columns (revenue, runtime, budget) to ensure no negative values"""
        df[self.numeric_columns] = df[self.numeric_columns].fillna(0)
        df[self.numeric_columns] = df[self.numeric_columns].astype(int)
        for col in self.numeric_columns:
            df = df[df[col]>=0]
        return df

    def remove_duplicates(self, df):
        """Remove duplicate entries based on title and release_date"""
        return df.drop_duplicates(subset=['title', 'release_year'])

    def clean_string_to_list(self, df):
        for col in self.string_columns:
            df[col] = df[col].fillna('')
            df[col] = df[col].str.split(", ")
            df[col] = df[col].apply(lambda x: [] if x == [''] else x)  # Make empty string list to empty list
        return df

    def clean_genre(self, genre: str) -> list[str]:
        # remove Film, film, Movies
        replacements = {
            r'\s*(Film|Movie|Films|Movies)$': '',  # Remove Film, Movie, Films, Movies at the end
            '/': ' ',
            'and ': '',
            '& ': '',
            "comdedy": "comedy",
            "documetary": "documentary",
            "-": " ",
            "animated": "animation",
            "biographical": "biography",
            "children's": 'children',
            "docudrama": "documentary drama",
            "educational": "education",
            "pornographic": "pornography",
            "sci fi": "scifi",
            "post apocalyptic": "postapocalyptic",
            " oriented": "",
            " themed": "",
            "fairy tale": "fairytale",
            "science fiction": "scifi",
        }

        # Apply regular expression replacements
        for pattern, replacement in replacements.items():
            genre = re.sub(pattern, replacement, genre, flags=re.IGNORECASE)

        # split multigenres
        if genre in genres_to_split:
            return genre.split(" ")
        else:
        # If they are not in the list of genres to split, then it's one genre and not mutilple
            return [] if genre == "" else [genre.strip()]



    def select_columns(self, df):
        return df[self.required_columns]

    def clean_dataset(self, input_path, output_path, sep=',', headers=[]):
        print('sep',sep)
        print('headers',headers)
        df = load_raw_data(input_path, sep, headers)
        print("original df shape", df.shape)
        if 'status' in df.columns:
            df = self.clean_status(df)
        print("after status", df.shape)
        if 'release_date' in df.columns:
            df = self.clean_release_date(df)
        print("after release date", df.shape)
        df = self.clean_release_year(df)
        print("after release year", df.shape)
        df = self.remove_duplicates(df)
        print("after duplicates", df.shape)
        df = self.clean_numeric_columns(df)
        print("after numeric columns", df.shape)
        df = self.clean_string_to_list(df)
        print("after string to list", df.shape)

        # clean_genres = []
        # for genres in df['genres']:
        #     clean_genres.extend(self.clean_movie_genres(genres))
        # df['genres'] = np.unique(clean_genres).tolist()
        # print("after genres", df.shape)

        df = self.select_columns(df)
        print("after select columns", df.shape)
        
        save_csv_data(df, output_path)
        return df


def main():
    TMDB_required_columns = ['title', 'release_date', 'revenue', 'runtime', 'budget', 'original_language', 'overview', 'genres',
            'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'release_year']

    TMDB_string_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
    TMDB_numeric_columns = ['revenue', 'runtime', 'budget']

    CMU_movie_headers = ["wikipedia_movie_id", "freebase_ID", "title", "release_year", "revenue",
                         "runtime", "languages", "countries",  "genres"]

    CMU_movie_required_columns_movie = ["wikipedia_movie_id",  "title", "release_year", "revenue", "runtime"] #, "genres"]
    CMU_string_columns_movie = [] #['genres']
    CMU_numeric_columns_movie = ['revenue', 'runtime']

    cleaner = DataCleaner(TMDB_required_columns, TMDB_string_columns, TMDB_numeric_columns)
    cleaner.clean_dataset('data/TMDB_movie_dataset_v11.csv', 'data/processed/TMDB_clean.csv')

    cleaner = DataCleaner(CMU_movie_required_columns_movie, CMU_string_columns_movie, CMU_numeric_columns_movie)
    cleaner.clean_dataset('data/raw/movie.metadata.tsv', 'data/processed/movies.csv', sep = '\t', headers = CMU_movie_headers)
    

if __name__ == "__main__":
    main()
