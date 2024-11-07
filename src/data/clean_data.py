import re
import os
import numpy as np
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path().resolve()  / 'src' / 'utils'))
from load_data import load_raw_data, save_csv_data

# ROOT_DIR = './' 
# DATA_DIR = os.path.join(ROOT_DIR, 'data')

# with open(os.path.join(DATA_DIR, 'genres_to_clean'), 'r') as file:
#     lines = file.readlines()

# # Remove newline characters if they exist at the end of each line
# genres_to_split = [line.strip() for line in lines]


class DataCleaner:
    def __init__(self):
        self.required_columns = [
            'title', 'status', 'release_date', 'revenue', 
            'runtime', 'budget', 'original_language', 'overview', 'genres',
            'production_companies', 'production_countries', 'spoken_languages', 'keywords']

        self.list_columns = [
            'genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords'
        ]
    
    def clean_status(self, df):
        """The movies must be released"""
        return df[df['status'] == 'Released']

    def select_columns(self, df):
        return df[self.required_columns]
    
    def clean_release_date(self, df):
        """Check if dates are in YYYY-MM-DD format"""
        
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_date'].dropna(inplace=True)
        return df
    
    def clean_numeric_columns(self, df):
        """Clean numeric columns (revenue, runtime, budget) to ensure no negative values"""
        numeric_columns = ['revenue', 'runtime', 'budget']
        df[numeric_columns] = df[numeric_columns].astype(int)
        for col in numeric_columns:
            df = df[df[col]>=0]
        return df

    def remove_duplicates(self, df):
        """Remove duplicate entries based on title and release_date"""
        return df.drop_duplicates(subset=['title', 'release_date'])


    def clean_string_to_list(self, df):
        for col in self.list_columns:
            df[col] = df[col].str.split(", ")
        return df
    
    def clean_dataset(self, input_path, output_path):
        df = load_raw_data(input_path)
        df = self.select_columns(df)
        print(f"inital size {df.shape}")
        df = self.clean_status(df)
        print(f"after cleaning status {df.shape}")
        df = self.clean_release_date(df)
        print(f"after cleaning release date {df.shape}")
        df = self.clean_numeric_columns(df)
        print(f"after cleaning numerical columns  {df.shape}")
        df = self.remove_duplicates(df)
        print(f"after removing duplicates {df.shape}")
        df = self.clean_string_to_list(df)
        
        save_csv_data(df, output_path)
        return df

    


# def clean_genre(genre: str) -> list[str]:
#     # remove Film, film, Movies
#     replacements = {
#         r'\s*(Film|Movie|Films|Movies)$': '',  # Remove Film, Movie, Films, Movies at the end
#         '/': ' ',
#         'and ': '',
#         '& ': '',
#         "comdedy": "comedy",
#         "documetary": "documentary",
#         "-": " ",
#         "animated": "animation",
#         "biographical": "biography",
#         "children's": 'children',
#         "docudrama": "documentary drama",
#         "educational": "education",
#         "pornographic": "pornography",
#         "sci fi": "scifi",
#         "post apocalyptic": "postapocalyptic",
#         " oriented": "",
#         " themed": "",
#         "fairy tale": "fairytale",
#         "science fiction": "scifi",
#     }

#     # Apply regular expression replacements
#     for pattern, replacement in replacements.items():
#         genre = re.sub(pattern, replacement, genre, flags=re.IGNORECASE)

#     # split multigenres
#     if genre in genres_to_split:
#         return genre.split(" ")
#     else:
#         return [] if genre == "" else [genre.strip()]

# def clean_movie_genres(genres: list[str]) -> list[str]:
#     """
#     Given list of genre names, return a clean version
#     """
#     clean = []
#     for genre in genres:
#          clean.extend(clean_genre(genre))

#     return np.unique(clean).tolist()



