import pandas as pd
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

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
    if (len(additional_stop_words) >0):
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