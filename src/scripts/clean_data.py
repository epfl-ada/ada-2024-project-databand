import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
sys.path.append(str(Path().resolve()  / 'src' / 'utils'))
from load_data import load_raw_data, save_csv_data

plot_headers = ["wikipedia_movie_id", "summary"]

import pdb 

class DataCleaner:
    def __init__(self, required_columns, string_columns, numeric_columns):
        self.required_columns = required_columns
        self.string_columns = string_columns
        self.numeric_columns = numeric_columns
    
    def clean_status(self, df):
        """The movies must be released"""
        return df[df['status'] == 'Released']
    
    def clean_adult(self, df):
        """Remove adult movies"""
        return df[df['adult'] == False]
    
    def clean_release_date(self, df):
        """Check if dates are in YYYY-MM-DD format"""
        
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_date'].dropna(inplace=True)
        df = df[df['release_date']>='1976-01-01'] # We want to keep movies from 1976 onwards, after the appearance of the VHS
        df = df[df['release_date'] < '2024-01-01']  # We want to keep movies until 2024
        return df
    
    def clean_release_year(self, df):
        if 'release_year' not in df.columns:
            df['release_year'] = df['release_date'].apply(lambda x: str(x)[:4] if pd.notnull(x) else None)
        
        df['release_year'] = df['release_year'].astype(str).str[:4]
        df = df[df['release_year']>='1976']
        df = df[df['release_year'] < '2024']
        return df

    def clean_numeric_columns(self, df):
        """Clean numeric columns (revenue, runtime, budget) to ensure no negative values"""
        df[self.numeric_columns] = df[self.numeric_columns].fillna(0)
        df[self.numeric_columns] = df[self.numeric_columns].astype(int)
        for col in self.numeric_columns:
            df = df[df[col]>=0]
        return df
    
    def clean_string_columns(self, df):
        """Clean string columns to lower case"""
        for col in self.string_columns:
            df[col] = df[col].str.lower()
        return df
    
    def clean_prod_companies(self, companies_list):
        if not isinstance(companies_list, list):
            return companies_list
        
        cleaned_companies = []
        for company in companies_list:
            if isinstance(company, str):
                cleaned = (company.strip()
                     .replace('films', '')
                     .replace('film', '')
                     .replace('pictures', '')
                     .replace('picture', '')
                     .replace('productions', '')
                     .replace('production', '')
                     .replace('entertainment', '')
                     .replace('media', '')
                     .replace('cinema', ''))
                cleaned = ' '.join(cleaned.split())
                cleaned_companies.append(cleaned.strip())
            else:
                cleaned_companies.append(company)
            
        return cleaned_companies


    
    def clean_runtime(self, df):
        """Remove movies that have a runtime between 45 & 500 minutes"""
        return df[((df['runtime'] <= 500) & (df['runtime'] > 45)) | (df['runtime'] == 0)]
    
    def remove_duplicates(self, df):
        """Remove duplicate entries based on title and release_date"""
        return df.drop_duplicates(subset=['title', 'release_year'])



    def select_columns(self, df):
        return df[self.required_columns]

    def clean_dataset(self, input_path, output_path, sep=',', headers=[]):
        df = load_raw_data(input_path, sep, headers)
        print("original df shape", df.shape)
        if 'status' in df.columns:
            df = self.clean_status(df)
            print("after status", df.shape)
        if 'adult' in df.columns:
            df = self.clean_adult(df)
            print("after adult", df.shape)
        if 'release_date' in df.columns:
            df = self.clean_release_date(df)
            print("after release date", df.shape)
        df = self.clean_release_year(df)
        print("after release year", df.shape)
        df = self.clean_runtime(df)
        print("after runtime", df.shape)
        df = self.remove_duplicates(df)
        print("after duplicates", df.shape)
        df = self.clean_numeric_columns(df)
        print("after numeric columns", df.shape)
        df = self.clean_string_columns(df)
        print("after string columns", df.shape)
        if 'production_companies' in df.columns:
            df['production_companies'] = df['production_companies'].apply(self.clean_prod_companies)
            print("after production companies", df.shape)
        df = self.select_columns(df)
        print("after select columns", df.shape)
        
        save_csv_data(df, output_path)
        return df


def main():
    TMDB_required_columns = ['title', 'release_date', 'revenue', 'runtime', 'budget', 'original_language', 'overview', 'genres',
            'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'release_year']

    TMDB_string_columns = ['title', 'genres', 'overview','production_companies', 'production_countries', 'spoken_languages', 'keywords']
    TMDB_numeric_columns = ['revenue', 'runtime', 'budget']

    CMU_movie_headers = ["wikipedia_movie_id", "freebase_ID", "title", "release_year", "revenue",
                         "runtime", "languages", "countries",  "genres"]

    CMU_movie_required_columns_movie = ["wikipedia_movie_id",  "title", "release_year", "revenue", "runtime"] #, "genres"]
    CMU_string_columns_movie = [] #['genres']
    CMU_numeric_columns_movie = ['revenue', 'runtime']

    cleaner = DataCleaner(TMDB_required_columns, TMDB_string_columns, TMDB_numeric_columns)
    cleaner.clean_dataset('data/raw/TMDB_movie_dataset_v11.csv', 'data/processed/TMDB_clean.csv')

    cleaner = DataCleaner(CMU_movie_required_columns_movie, CMU_string_columns_movie, CMU_numeric_columns_movie)
    cleaner.clean_dataset('data/raw/movie.metadata.tsv', 'data/processed/movies.csv', sep = '\t', headers = CMU_movie_headers)

    df = load_raw_data('data/raw/plot_summaries.txt', sep='\t', headers=['wikipedia_movie_id', 'summary'])
    save_csv_data(df, 'data/processed/plot_summaries.csv')

if __name__ == "__main__":
    main()
