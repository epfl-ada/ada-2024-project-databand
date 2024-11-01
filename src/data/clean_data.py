import re
import os
import numpy as np
import pandas as pd

current_folder = os.path.dirname(__file__)
data_folder = os.path.join(current_folder, '../../data')
data_file_path = os.path.join(data_folder, 'genres_to_clean.txt')

with open(data_file_path, 'r') as file:
    lines = file.readlines()

# Remove newline characters if they exist at the end of each line
genres_to_split = [line.strip() for line in lines]

def clean_genre(genre: str) -> list[str]:
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
        return [] if genre == "" else [genre.strip()]

def clean_movie_genres(genres: list[str]) -> list[str]:
    """
    Given list of genre names, return a clean version
    """
    clean = []
    for genre in genres:
         clean.extend(clean_genre(genre))

    return np.unique(clean).tolist()

def get_date_year(date):
    # Attempt to parse the date in 'YYYY-MM-DD' format
    year = pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')

    if pd.notna(year):
        return year.year

    # Attempt to parse in 'YYYY' format
    year = pd.to_datetime(date, format='%Y', errors='coerce')

    if pd.notna(year):
        return year.year

    return None

def clean_language(lang):
    '''
    Create a dict mapping to replace some languages in the data to correct version
    e.g. France -> French
    '''
    pass

