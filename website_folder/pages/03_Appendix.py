import streamlit as st
from pathlib import Path
import sys

# Add project root to Python path
root_dir = Path(__file__).parent.parent.parent  # Go up two levels from the current file
sys.path.append(str(root_dir))

from src.data.process_data import create_cmu_tmdb_dataset
from src.utils.website_utils import *
import json
from src.utils.data_utils import get_proportions, categorize_production
# from src.models.empath_model import *

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

prop_rolling_avg = budget_rolling_averages(df_filtered, window=3)

# Define categories and their properties
categories = [
    ('Independent', 'Independent productions (<0.1x mean budget)', '#9B59B6'),
    ('Small', 'Small productions (<1x mean budget)', '#E74C3C'),
    ('Big', 'Big productions (>1x mean budget)', '#28B463'),
    ('Super', 'Super productions (>5x mean budget)', '#2E86C1')
]

# Create and display the combined plot
fig = create_combined_plot(prop_rolling_avg, categories)

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

st.markdown("Regarding the production type proportions for major regions, the CMU dataset paints a steadier picture of how DVDs influenced global movie production compared to TMDB's more dynamic trends. In Eastern Asia, independent films maintained their dominance across eras, with only a modest rise in super productions during the DVD era, contrasting TMDB's dramatic shifts. North America stands out as the stronghold for super productions, which dominated all eras in the CMU data, unlike TMDB's notable post-DVD rise in independent films. Europe shows some similarity to TMDB, with a decline in independent movies during the DVD era, but CMU suggests this shift was more sustained and less temporary. Interestingly, Oceania, absent from TMDB, mirrors Europe's patterns, further emphasizing the global foothold of big and super productions. Overall, the CMU data suggests a smoother and more consistent adoption of production trends across regions, lacking the volatility seen in TMDB’s analysis.")

countries_genres_props = get_proportions(df_countries_filtered, ['dvd_era', 'region'], 'genres')
fig = return_num_companies_per_prod_type(df)
st.plotly_chart(fig)

st.markdown("The rise of DVDs not only transformed distribution but also reshaped how production companies collaborated. Both the CMU and TMDB graphs highlight increasing collaboration among production companies, particularly for Super productions post-DVD. However, CMU shows steadier trends, with Big and Small productions gradually integrating into collaborative networks, compared to TMDB’s more erratic patterns. Independent productions remain isolated in both datasets, with minimal growth in collaboration. While dominant players likely lead in both, CMU suggests a slightly less centralized network compared to TMDB’s concentrated collaboration among major giants.")

df_genres = df[df.genres.apply(lambda x: len(x) > 0)].copy()
df_genres = df_genres.explode('genres')
df_genres.head()
df_genres = df_genres[df_genres.budget > 0]
genre_proportions = get_proportions(df_genres, base_vars=['prod_type', 'dvd_era'], target_var='genres')

fig = return_genre_trends_by_prod_type(genre_proportions)

st.plotly_chart(fig)

st.markdown("The CMU dataset reveals a more genre-specific evolution across production types during the DVD era compared to the TMDB dataset's broader narrative. For independent productions, Drama saw significant growth in both datasets, but CMU highlights Thriller as another genre on the rise, whereas TMDB emphasized Comedy and Horror. Small productions maintained a steady trajectory, with Drama experiencing growth in both datasets, though CMU points to a more stable overall landscape. Big productions showed remarkable consistency in the CMU data, with only a minor boost for Thriller, aligning with TMDB’s emphasis on stability for this production type. Super productions, however, diverge noticeably: while TMDB focuses on Action and Animation taking the lead, CMU showcases substantial growth in Adventure, alongside Fantasy and Family, suggesting a shift toward genres with strong imaginative appeal. Both datasets agree on the transformative impact of DVDs, yet CMU paints a picture of subtle, focused changes rather than the broader shifts emphasized in TMDB.")



results = pd.read_csv('data/website_data/CMU/topics_per_genre_prod.csv')

# Add a dropdown (selectbox) for genre selection
genre_options = results['genre'].unique()
selected_genre = st.selectbox("Select Genre:", genre_options)

# Plot the data based on the selected genre
fig_empath = plot_all_features(results, selected_genre)

# Display the plot
st.pyplot(fig_empath)


st.markdown("Similarly to the TMDB dataset, the drama movies in the CMU dataset also emphasize themes of family and community, but the trends show distinct differences across production types. Independent dramas initially focus on themes like family and home, with slight growth in positive emotion, while themes like war and crime see a decline during the DVD era. For Small productions, family and celebration emerge as central themes, with noticeable growth during the DVD years, reinforcing their universal appeal. Big productions maintain a consistent focus on family and war themes, showing stability rather than dramatic shifts. Meanwhile, Super productions take a different turn, with themes like war, sports, and leadership dominating in the pre-DVD era but declining sharply during the DVD period as more diverse themes like friendship and competing gained traction. This shift indicates a movement toward narratives with broader audience appeal, even among high-budget dramas.")

fig = return_movies_prop_per_region(region_props)
st.plotly_chart(fig)

st.markdown("This shift in thematic focus for drama productions across different scales raises an important question: how do these evolving storylines align with the global trends in movie production? To answer that, we turn our attention to the regional dynamics of movie releases over time, examining how the rise of DVDs influenced the proportion of films emerging from different parts of the world. Between the pre-DVD era and the peak DVD era, Western Asia shows a significant decline in its proportion of movie releases, dropping sharply after its dominance in earlier years. In contrast, regions like North America and Europe maintain relatively stable proportions, showing slight growth post-2000. Eastern Asia steadily rises across the DVD era, indicating a growing influence in global cinema. Other regions, such as South America, Africa, and Southeast Asia, remain relatively flat, reflecting limited change in their global share of movie releases during this period. This highlights the growing prominence of Eastern Asia and the sustained dominance of established regions like North America and Europe during the DVD era.")

fig = return_genre_prop_per_region(selected_regions, countries_genres_props)
st.plotly_chart(fig)

st.markdown("The CMU dataset provides a nuanced view of global genre trends during the DVD era, offering both overlap and contrast with TMDB's analysis. While Drama remains the dominant genre across most regions, the CMU data highlights regional differences: in Eastern Asia, Drama and History strongly rises as family genres decrease, diverging from TMDB's trend. Europe, UK and North America show steady Drama growth, while North America sees adventure gaining traction, reflecting super production influence more clearly than TMDB. Regarding Southern Asia and Oceania, their trends are very similar to the TMDB, focusing on drama and its decrease. Overall, CMU emphasizes broader regional diversification compared to TMDB’s global trends, while still aligning on Drama's global strength.")

# Style for the Back button (bottom left)
st.markdown("""
    <style>
        /* Style the Back button for both light and dark mode */
        .stBackButton button {
            position: fixed;
            bottom: 40px;
            left: 40px;
            background-color: rgba(255, 87, 34, 0.8); /* A nice orange with transparency */
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 20px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.06); /* Subtle shadow */
            transition: all 0.3s ease;
        }

        .stBackButton button:hover {
            background-color: rgba(255, 87, 34, 1); /* Brighter orange on hover */
            transform: scale(1.05); /* Slight zoom effect */
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2), 0 2px 4px rgba(0, 0, 0, 0.1); /* Enhanced shadow */
        }

        /* Ensure compatibility with both light and dark themes */
        @media (prefers-color-scheme: light) {
            .stBackButton button {
                background-color: rgba(255, 87, 34, 0.9); /* Slightly darker orange for light mode */
                color: white;
            }

            .stBackButton button:hover {
                background-color: rgba(255, 69, 0, 1); /* Complementary hover effect */
            }
        }

        @media (prefers-color-scheme: dark) {
            .stBackButton button {
                background-color: rgba(255, 87, 34, 0.9); /* Bright orange for dark mode */
                color: white;
            }

            .stBackButton button:hover {
                background-color: rgba(255, 69, 0, 1); /* Complementary hover effect */
            }
        }
    </style>
""", unsafe_allow_html=True)

# Add the Back button
if st.button("← Analysis"):
    st.switch_page("pages/02_Analysis.py")

st.markdown("---")
st.write("Created with ❤️ using Streamlit")