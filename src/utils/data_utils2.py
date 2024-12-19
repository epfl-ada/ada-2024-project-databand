import pandas as pd
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
import re

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stop_words.update(['film', 'find'])

def top_proportions_per_era(df, column, k, islist=False):
    # get total number movies grouped by era and column
    if islist:
        counts = df.explode(column).groupby(['dvd_era', column]).size().reset_index(name='count')
    else:
        counts = df.groupby(['dvd_era', column]).size().reset_index(name='count')

    # get proportion
    total_counts = counts.groupby('dvd_era')['count'].transform('sum')
    counts['proportion'] = counts['count'] / total_counts

    # get top-k for each era
    top_k = counts.sort_values(by=['proportion'], ascending=[False]).groupby('dvd_era').head(k)
    category_order = ['pre', 'during', 'post']
    top_k['dvd_era'] = pd.Categorical(top_k['dvd_era'], categories=category_order, ordered=True)

    return top_k

def create_wordcloud(text, additional_stop_words):
    if (len(additional_stop_words) > 0):
        new_stop_words = set(stopwords.words("english")).copy()
        new_stop_words.update(additional_stop_words)
    else:
        new_stop_words = stop_words
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=new_stop_words,
        max_words=100
    ).generate(text)

    return wordcloud


def wordcloud_per_genre(df, genre, additional_stop_words):
    wordclouds = []
    for era in ['pre', 'during', 'post']:
        year_texts = df[(df['genres'].apply(lambda x: genre in x)) & (df['dvd_era'] == era)]['overview'].str.cat(
            sep=' ')

        if not year_texts:
            print(f"No movies found")
            return

        # Create wordcloud
        wordcloud = create_wordcloud(year_texts, additional_stop_words)
        wordclouds.append(wordcloud)
    return wordclouds

def get_movies_for_genre(df, genre):
    return df[df['genres'].apply(lambda x: genre in x)]

def get_movie_plots(df, genre, era):
    df_genre = get_movies_for_genre(df, genre)
    text_data = df_genre[(df_genre['dvd_era'] == era) & (df_genre['clean_overview'])]['clean_overview'].apply(
        lambda x: str(x)).explode().tolist()
    return text_data

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = re.sub(r'\W+', ' ', text)
    return " ".join([word for word in str(text).split() if word not in stop_words])


def adjust_inflation(df, old_col="budget", new_col="budget_inflation", inflation_file="data/cpi_inflation.csv"):
    """
    Adjusts the budget of movies for inflation.
    """
    # Read the inflation file
    inflation = pd.read_csv(inflation_file)

    # Calculate the cumulative inflation factor
    inflation['Cumulative_Inflation'] = (1 + inflation['CPALTT01USM657N'] / 100).cumprod()

    # Extract the year from the date
    inflation['Year'] = pd.to_datetime(inflation['DATE']).dt.year

    # Create a dictionary for inflation factors
    inflation_dict = inflation.set_index('Year')['Cumulative_Inflation'].to_dict()

    # Adjust the budget for inflation
    df[new_col] = df.apply(lambda row: row[old_col] * inflation_dict.get(row['release_year'], 1), axis=1)

    return df

def categorize_production(row, means):
    mean_budget = means.loc[row.release_year, 'mean_budget']
    if row['budget'] < 0.1 * mean_budget:
        return 'Independent'
    elif row['budget'] < mean_budget:
        return 'Small'
    elif row['budget'] < 5 * mean_budget:
        return 'Big'
    else:
        return 'Super'

def categorize_budget(row, budget_stats):
    mean_budget_row = budget_stats.loc[budget_stats['release_year'] == row['release_year'], 'mean_budget']
    if mean_budget_row.empty:
        return 'Unknown'
    mean_budget = mean_budget_row.values[0]
    if row['budget'] < 0.1 * mean_budget:
        return 'Independent'
    elif row['budget'] < mean_budget:
        return 'Small'
    elif row['budget'] < 5 * mean_budget:
        return 'Big'
    else:
        return 'Super'

def budget_rolling_averages(df, window):
    budget_stats = df.groupby('release_year')['budget'].agg(mean_budget='mean').reset_index()
    df.loc[:, 'budget_category'] = df.apply(categorize_budget, args=(budget_stats,), axis=1)

    # Count the number of each budget category per year
    budget_category_counts = df.groupby(['release_year', 'budget_category']).size().unstack(fill_value=0)

    # Calculate the proportion of each budget category per year
    budget_category_proportions = budget_category_counts.div(budget_category_counts.sum(axis=1), axis=0)

    # Calculate the 3-year rolling average for each budget category
    proportion_rolling = budget_category_proportions.rolling(window=window, center=True).mean()
    return proportion_rolling


def calculate_roi(df):
    if df.revenue > 0 and df.budget > 0:
        return (df.revenue - df.budget) / df.budget * 100
    else:
        return 0


def get_proportions(df, base_vars, target_var):
    counts = df.groupby(base_vars + [target_var], observed=False).size().reset_index(name='count')
    totals = df.groupby(base_vars, observed=False).size().reset_index(name='total')
    props = counts.merge(totals, on=base_vars)
    props['prop'] = props['count'] / props['total']
    return props



