import streamlit as st
from src.data.process_data import create_cmu_tmdb_dataset
from src.utils.website_utils import *
import json
from src.utils.data_utils import get_proportions, categorize_production
from src.models.empath_model import *

st.set_page_config(
    page_title="Appendix",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Data loading

df = create_cmu_tmdb_dataset(cmu_movies_path='data/processed/movies.csv', plots_path='data/processed/plot_summaries.csv', 
                             tmdb_path='data/processed/TMDB_clean.csv', how_merge='inner')
df['log_revenue'] = np.log10(df['revenue'].replace(0, np.nan))
df_filtered = df[df['revenue'] > 0]

mean_budgets = df[df.budget > 0].groupby('release_year').agg(mean_budget=('budget', 'mean'))
df['prod_type'] = df.apply(categorize_production, axis=1, args=(mean_budgets,))
df = df[df['dvd_era'].isin(['pre', 'during'])]
df['dvd_era'] = pd.Categorical(df['dvd_era'], categories=['pre', 'during'], ordered=True)
df['prod_type'] = pd.Categorical(df['prod_type'], categories=['Independent', 'Small', 'Big', 'Super'], ordered=True)

# Title and Introduction
st.title("Additionnal plots from CMU dataset")

st.markdown("Our original dataset was the CMU Movie Summary Corpus, which unfortunately lacked some key features such as budget, revenue, or production countries, and only contained a few movies from the post-DVD era, pushing us to turn to the TMDB dataset. However, we performed a subset of our analyses on this dataset, focused on examining the impact of the rise of DVDs, leveraging information from the TMDB dataset. Let’s see if we reach the same conclusions!")

st.header("Revenue")

st.markdown("Similar to the TMDB dataset, we observe that DVDs lead to a broader range of revenues, with more movies generating either low-revenue or quite high-revenue.  ")



#- **Recovery Opportunities**: Underperforming films could achieve financial success through strong DVD sales.
#- **Broader Accessibility**: DVDs expanded the reach of films, particularly benefiting family-friendly and niche genres.

fig_revenue = create_histogram(
df_filtered,
x_col='log_revenue',
color_col='dvd_era',
title='Revenue Distribution Across DVD Eras',
labels={'log_revenue': 'Revenue (log scale)', 'dvd_era': 'DVD Era'},
color_palette=['#2E86C1', '#28B463', '#E74C3C']
)

st.plotly_chart(fig_revenue, use_container_width=True)

st.markdown("For budgets however, the picture is a little grimmer for Independent productions here. While they showed a massive popularity boost with the entire TMDB dataset, there isn’t much of a difference in proportion over time here. Instead, small and big productions maintain their foothold.  ")

fig = return_budget_rolling_averages(df)

st.plotly_chart(fig, use_container_width=True)

st.markdown("By examining the impact on different world regions once again, we indeed see that Small and Big productions are the most prominent in most regions. The only exception is Southern Asia, where Independent production movies are surprisingly common, even pre-DVD era.  ")

with open('./data/countries_to_region.json', 'r') as file:
    countries_regions = json.loads(file.read())
    
df_countries = df.copy().explode('production_countries')
df_countries = df_countries.explode('genres')
df_countries['region'] = df_countries.production_countries.apply(lambda x: countries_regions[x] if x in countries_regions and pd.notna(x) else None)
df_countries.dropna(subset=['region'], inplace=True)

region_props = get_proportions(df_countries, ['release_year'], 'region')


selected_regions = list(region_props[region_props.prop > 0.05].region.unique())
df_countries_filtered = df_countries[(df_countries.region.isin(selected_regions))
                                     & (df_countries.budget > 0)]

fig = return_prod_type_prop_per_region(selected_regions, df_countries_filtered)
st.plotly_chart(fig)


countries_genres_props = get_proportions(df_countries_filtered, ['dvd_era', 'region'], 'genres')



fig = return_num_companies_per_prod_type(df)
st.plotly_chart(fig)

df_genres = df[df.genres.apply(lambda x: len(x) > 0)].copy()
df_genres = df_genres.explode('genres')
df_genres.head()
df_genres = df_genres[df_genres.budget > 0]
genre_proportions = get_proportions(df_genres, base_vars=['prod_type', 'dvd_era'], target_var='genres')

fig = return_genre_trends_by_prod_type(genre_proportions)

st.plotly_chart(fig)

empathModel = EmpathModel()

results = pd.read_csv('data/website_data/CMU/topics_per_genre_prod.csv')

empathModel.plot_all_features(results, 'drama')

# Display the Empath model results
fig_empath = empathModel.plot_all_features(results, 'drama')
st.pyplot(fig_empath)

fig = return_movies_prop_per_region(region_props)
st.plotly_chart(fig)


fig = return_genre_prop_per_region(selected_regions, countries_genres_props)
st.plotly_chart(fig)