import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
import json
import geopandas as gpd
from plotly.subplots import make_subplots

root_dir = Path(__file__).parent.parent.parent  # Go up two levels from the current file
sys.path.append(str(root_dir))

from src.data.process_data import create_tmdb_dataset
# from src.utils.load_data import load_raw_data
from src.utils.data_utils import *
from src.utils.plot_utils import *
from src.utils.website_utils import *

def load_region_mapping(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

DATA_PATH = Path("data")
TMDB_PATH = DATA_PATH / "processed/TMDB_clean.csv"
REGION_MAP_PATH = DATA_PATH / "countries_to_region.json"


df = create_tmdb_dataset(TMDB_PATH)
region_mapping = load_region_mapping(REGION_MAP_PATH)

df['log_revenue'] = np.log10(df['revenue'].replace(0, np.nan))
df_filtered = df[df['revenue'] > 0]

st.set_page_config(
    page_title="Was Matt Damon right?",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and Introduction
st.title("Is Matt Damon Right? Did the Fall of DVDs Doom the Film Industry?")
st.image(
    "website_folder/pages/Why-are-we-surprised-by-Matt-Damons-recent-toxic-comments.webp",  
    caption="Matt <3", 
    use_container_width=True  # Adjust the image size to fit the column width
)

st.markdown("""
The film industry has undergone seismic shifts over the past few decades, driven largely by changes in distribution models. Among the most notable innovations was the emergence of **DVDs in the 1990s**. DVDs revolutionized the accessibility of movies, providing a new revenue stream for production companies and reshaping both the business and creative landscapes of filmmaking. 

This project explores these transformations through an analysis of key differences observed in data from various eras of the film industry, focusing on:
- Revenue
- Budgets
- Production dynamics
- Genre evolution
""")

st.markdown("---")

# Revenue Section
st.header("A Changing Financial Landscape")
st.subheader("Leveling the Playing Field")
st.markdown("""
The introduction of DVDs provided a secondary revenue stream for production companies. Many movies that struggled at the box office found profitability in home entertainment. This diversification of revenue sources allowed for:

This visualization provides a look at how the revenue of movies has evolved over three distinct eras in the film industry: pre-DVD, during the DVD era, and post-DVD, with their revenue displayed on a logarithmic scale.
""")
st.write("")
col1, col2 = st.columns([1, 1])
#- **Recovery Opportunities**: Underperforming films could achieve financial success through strong DVD sales.
#- **Broader Accessibility**: DVDs expanded the reach of films, particularly benefiting family-friendly and niche genres.
with col1:
    fig_revenue = create_histogram(
        df_filtered,
        x_col='log_revenue',
    color_col='dvd_era',
    title='Revenue Distribution Across DVD Eras',
    labels={'log_revenue': 'Revenue (log scale)', 'dvd_era': 'DVD Era'},
    color_palette=['#2E86C1', '#28B463', '#E74C3C']
)
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    st.markdown(""" 
**The Pre-DVD Era (Red):** Before the rise of DVDs, the film industry was dominated by theatrical releases, with a significant number of movies achieving mid-to-high revenue levels (log revenue scale of 5–7, equivalent to 100,000-10,000,000 dollars). 
""")
    st.markdown("""
**The DVD Era (Blue):** The introduction of DVDs changed the industry. This era saw a sharp increase in the number of movies generating moderate revenue (log revenue scale of 3–6, roughly 1,000-1,000,000 dollars). DVDs provided smaller-budget and niche films with an alternate revenue stream, democratizing the market and allowing more films to find financial success even if they didn’t dominate at the box office.
""")
    st.markdown(""" 
**The Post-DVD Era (Green):** With the decline of DVDs and the rise of streaming platforms, the revenue landscape shifted once again. The distribution reveals a drop in high-revenue films, with fewer movies achieving blockbuster-level revenues (log revenue scale above 7). The majority of movies now cluster in the lower revenue ranges (log revenue scale of 3–6), reflecting the challenges faced by filmmakers in the subscription-driven streaming era.
""")
st.write("")
with st.expander("📝 Important Note", expanded=True):
    st.info(
        """
        It is important to note that statistical tests did not reveal significant differences 
        in revenue distributions between DVD eras. However, our plot still paints an interesting 
        picture: movies released during the peak DVD era show a broader revenue range and a 
        slight tilt toward higher earnings compared to pre- and post-DVD eras. It seems that 
        DVDs really opened up some new opportunities for studios to make more money!
        """
    )

# Budget Section
st.header("Budget Dynamics: Empowerment Through DVDs")
st.markdown("""
By providing an alternate revenue stream, the rise of DVDs at their peak allowed smaller production companies to enter the market.
Indeed, small studios could directly distribute their movies as DVDs, without the hassle and expense of going for a theatrical release. 
""")
st.markdown("""
To find evidence of this trend, we define 4 types of production companies using the mean budget as reference:
- **Independent** (<0.1x mean)
- **Small** (<1x mean)
- **Big** (>1x mean)
- **Super** (>5x mean)

Let’s take a look at how these productions evolve over time:
""")
# Calculate rolling averages
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
# fig = return_budget_rolling_averages(df)

# st.plotly_chart(fig, use_container_width=True)
st.markdown("""
From the plot, we can see a clear rise in movies from independent production companies after DVDs hit the scene, with small productions maintaining a strong presence.

However, it seems that the fall of DVDs reshaped budget allocations once again… 

While Small production companies remained consistently prevalent during the DVD era, their popularity took a slight hit when DVD sales started declining. It seems that, without a viable secondary market, these projects struggled to find footing. 
In contrast, in peak DVD era was also marked with a slight boost in Super production companies, which maintained their influence in the movie industry after DVDs’ downfall. Indeed, some studios focused on large-scale productions with global earning potential.
""")

with st.expander("🔍 Key Observations", expanded=True):
    row1_col1, row1_col2 = st.columns([1, 1])
    row2_col1, row2_col2 = st.columns([1, 1])

    # Fill the grid with content
    with row1_col1:
        st.info("""
            **1. Independent Productions**
            
            Independent films have seen significant growth, rising from 40–50% in 1980–1995 to nearly 70% by 2020. The DVD era (1995–2005) likely encouraged independent productions by providing an accessible distribution channel, allowing smaller studios to bypass expensive theatrical releases. The rise of streaming platforms like Netflix and Amazon post-2010 also created a surge in demand for low-budget content.
        """)

    with row1_col2:
        st.info("""
            **2. Small Productions**
            
            Peaking at 30% during the DVD era (1990s–2005), small productions have declined sharply, making up less than 10% of movies by 2020. DVDs allowed small-budget films to target niche audiences, but streaming platforms now favor either low-budget independents or high-budget blockbusters.


        """)

    with row2_col1:
        st.info("""
            **3. Big Productions**
            
            Once over 35% of movies in the 1980s and 1990s, big productions have steadily declined, dropping below 15% by 2020. The decline of DVDs reduced big/mid-budget films' revenue streams, and streaming platforms prioritize independent or blockbuster content. The shrinking space for mid-budget films reflects a shift to extremes in the industry's financial model.


        """)

    with row2_col2:
        st.info("""
            **4. Super Productions**
            
            Super productions have grown steadily since the 1980s, accounting for over 6% of movies by 2020, with rapid growth post-2000. Studios shifted to blockbusters to maximize theatrical revenue as DVDs declined, and in the streaming era, blockbusters drive subscriptions and dominate global viewership. The rise of super productions underscores the industry's focus on fewer, high-budget films with global market appeal.
        """)

st.write("""
Let’s summarize all this in a plot. The film industry has seen a shift in production trends, with independent films and super productions growing in prominence, while small and mid-budget films have declined. 
""")
mean_budgets = df[df.budget > 0].groupby('release_year').agg(mean_budget = ('budget', 'mean'))
df['prod_type'] = df.apply(categorize_production, axis=1, args=(mean_budgets,))
df['dvd_era'] = pd.Categorical(df['dvd_era'], categories=['pre', 'during', 'post'], ordered=True)
df['prod_type'] = pd.Categorical(df['prod_type'], categories=['Independent', 'Small', 'Big', 'Super'], ordered=True)


df_filtered = df[df.budget > 0]
props = (df_filtered.groupby('dvd_era')['prod_type']
         .value_counts(normalize=True)
         .unstack())

# Production type proportions across DVD eras
fig = create_stacked_bar(props)
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.subheader("Was This a Global Phenomenon?")
st.markdown("""Let's start by focusing on the production countries of our movies. Since there are many, we grouped them into main global regions:  
""")

# World Map visualization
try:
    with open('data/countries_to_region.json', 'r') as file:
        countries_regions = json.load(file)
    
    fig, error = create_world_map(countries_regions, 'data/ne_110m_admin_0_countries.shp')
    
    if error:
        st.error(error)
        st.write("Please ensure the shapefile is in the correct location: data/ne_110m_admin_0_countries.shp")
    else:
        st.plotly_chart(fig, use_container_width=True)
        
except FileNotFoundError:
    st.error("Could not find the countries_to_region.json file. Please ensure it's in the data directory.")

st.markdown("""Now that we have sorted our countries into broader regions, let's take a look at who are the biggest players, i.e. which regions produce the most movies.""")

# Calculate region proportions using the new data processing
df_countries = df.copy().explode('production_countries')
df_countries = df_countries.explode('genres')
df_countries['region'] = df_countries.production_countries.apply(
    lambda x: countries_regions[x] if x in countries_regions and pd.notna(x) else None
)
df_countries.dropna(subset=['region'], inplace=True)

region_counts = df_countries.groupby(['release_year', 'region']).size().reset_index(name='count')
filtered_regions = (region_counts.groupby('region')
                   .sum('count')
                   .sort_values(by='count', ascending=False)
                   .reset_index()
                   .head(16)
                   .region.to_list())

total_counts = df_countries.groupby(['release_year']).size().reset_index(name='total')
region_prop = region_counts.merge(total_counts, on='release_year')
region_prop['prop'] = region_prop['count'] / region_prop['total']

# Create and display the regional distribution plot
fig = create_regional_distribution_plot(region_prop)
plot_col, text_col = st.columns([3, 1]) 
with plot_col:
    st.plotly_chart(fig, use_container_width=True)
with text_col:
    with st.expander("📊 Regional Trends Analysis", expanded=False):
        st.info("""
            Interestingly, Eastern Asia and Europe show opposite trends in movie releases around the DVD era: 
            Eastern Asian releases dipped slightly during this time, while European releases climbed. Meanwhile, 
            North American movies, which have dominated since the mid-80s, hit their golden era in the late 1990s—just 
            as DVDs emerged—and have seen a small but steady decline ever since.
        """)
st.markdown("""Let's focus on the biggest film-producing regions, and analyze their composition of production types:""")

# Filter for major regions
selected_regions = list(region_prop[region_prop.prop > 0.05].region.unique())
df_countries_filtered = df_countries[(df_countries.region.isin(selected_regions)) 
                                   & (df_countries.budget > 0)]

# Create and display the regional production subplots
fig = create_regional_production_subplots(df_countries_filtered, selected_regions)
st.plotly_chart(fig, use_container_width=True)

st.write("""Clearly, DVDs shook up the movie industry in different ways across world regions! Even among our major players—Eastern Asia, Europe, and North America—the trends vary widely. In Eastern Asia, for example, independent movies were already a staple in the pre-DVD era but gave way to super productions once DVDs arrived. In North America, super productions also gained traction post-DVD, but this came hand-in-hand with a significant rise in independent films, mirroring the global trend. Europe, however, stands out with the most surprising shift: independent movies declined, while super productions rose—but only during the DVD era!""")

st.header("Production Shifts: From Independence to Streaming Giants")

# The Rise of Independent Studios
st.subheader("The Rise of Independent Studios")
st.markdown("""
During the DVD era, independent studios leveraged home entertainment to:
- **Distribute Niche Films**: DVDs offered a cost-effective means to reach audiences.
- **Challenge Major Studios**: Smaller players gained prominence in a market previously dominated by blockbusters.
""")

# Streaming’s Market Takeover
st.subheader("Streaming’s Market Takeover")
st.markdown("""
With the decline of physical media, new players reshaped the production landscape:
- **Dominance of Streaming Services**: Companies like Netflix and Amazon emerged as major content producers.
- **Traditional Studios Adapt**: Legacy studios faced consolidation and strategic realignments to compete in the digital age.
""")









# Get mean number of production companies per year
yearly_avg_companies = (df[df['production_companies'].str.len() > 0].groupby('release_year')
                       .agg({'production_companies': lambda x: x.str.len().mean()})
                       .reset_index())

st.markdown("""
So, DVDs allowed new players to enter the market, especially for low-budget movies. Interestingly, there’s also been an increase in collaboration between different production companies over time, as evaluated by the number of companies that produced a given movie:

Indeed, we obtain a strong correlation between the number of production companies per movie and the release year. However, let’s check if this is really the case for all production types:
""")

# Create two columns for the plots
col1, col2 = st.columns([1, 1])

# Call the function to get the plots
fig, heatmap_fig = return_avg_num_prod_companies(yearly_avg_companies)

# Display the first plot (average number of production companies) in the first column
with col1:
    st.plotly_chart(fig)

# Display the second plot (correlation heatmap) in the second column
with col2:
    st.plotly_chart(heatmap_fig)

# Create columns
col1, col2 = st.columns([1, 1])

# Generate the figure from the function
fig = return_num_companies_per_prod_type(df)

# Display the text in the first column
with col1:
    st.markdown("""
    It looks like bad news for independent productions… Indeed, while we observe a strengthening collaboration between production companies for movies of most production types after the rise of DVDs, companies of low-budget films appear to be left behind.
    """)

# Display the plot in the second column
with col2:
    st.plotly_chart(fig)

st.markdown("""But who’s collaborating with whom? We create a network of our companies, linking those that co-produced a movie at least once. The result is 5 clusters, with a clear dominant one. If we take a closer look at who are the major collaborators, we recognize some familiar names - “Paramount”, “20th Century Fox”, or “Columbia Pictures”.  The dominance of these major players in the collaboration space suggests that while the market has become more accessible, true influence in the industry remains concentrated among established giants.""")

df_graph = df[df['production_companies'].str.len() > 0]

before_DVD_era_super = df_graph[(df_graph['dvd_era'] == 'pre') & (df_graph['prod_type'] == 'Super')]

fig = return_collaborations(before_DVD_era_super)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.header("Genre Evolution: Reflecting Changing Preferences")

# select rows with at least one genre entry
df_genres = df[df.genres.apply(lambda x: len(x) > 0)].copy()

# explode genres list
df_genres = df_genres.explode('genres')

# get proportion of movies per genre, for each prod_type-dvd_era pair
genre_proportions = get_proportions(df_genres, base_vars=['prod_type', 'dvd_era'], target_var='genres')

# since we look at production types, only consider movies that have a budget > 0
df_genres = df_genres[df_genres.budget > 0]

# Layout with columns for explanatory text and plot
colImp1, colImp2 = st.columns([1, 2])

# Column 1: Explanatory text
with colImp1:
    st.subheader("The Impact of DVDs on Genre Trends")
    st.markdown("""
    The advent and rise of DVDs transformed the movie industry, reshaping genre trends across different production types while responding to changing audience preferences.

    First, let’s check the most popular genres for each production type:

    For independent, small, and big production movies, we observe a clear preference for **drama**, **comedy**, and **thriller** movies. In contrast, **super production** movies appear to favour **action** and **adventure** movies. By analyzing the genre preferences across DVD eras, we observe that most remained relatively stable over time:  
    """)

# Column 2: First Plot - Genre Proportions by Production Type
with colImp2:
    figImp = return_genre_prop_by_prod_type(genre_proportions)
    st.plotly_chart(figImp, use_container_width=True)

# Layout with columns for explanatory text and second plot
colTr3, colTr4 = st.columns([1, 2])

# Column 3: Explanatory text for the second plot
with colTr3:
    st.subheader("Genre Trends Across DVD Eras")
    st.markdown("""
    By examining the trends of different genres across the DVD eras, we can see how production types influenced genre evolution.
    While genres like **drama** and **comedy** remain popular across all eras, the **action** and **adventure** genres see more prominence in super productions. The **thriller** genre also gained momentum during the DVD era, with its appeal to a broad audience.  
    """)

# Column 4: Second Plot - Genre Trends by Production Type
with colTr4:
    figTr = return_genre_trends_by_prod_type(genre_proportions)
    st.plotly_chart(figTr, use_container_width=True)


# Displaying the insights and trends in Streamlit using Markdown
st.header("Insights and Trends")

# Independent Production Type
st.markdown("#### 1. Independent Production Type")
st.markdown("""
- **Drama and Comedy** remain dominant genres across all eras, with relatively consistent proportions.
- **Horror** shows a slight increase during the DVD era, likely influenced by the rise of home video markets that cater to niche audiences.
- **Family and Animation genres** remain underrepresented, likely due to the higher production costs typically required for these genres, which independent productions struggle to afford.
""")

# Small Production Type
st.markdown("#### 2. Small Production Type")
st.markdown("""
- **Comedy and Drama** dominate, but **Action and Adventure** show slight growth post-DVD, likely driven by increasing audience expectations for higher production values, even in smaller-scale movies.
- **Family movies** saw a noticeable increase during the DVD era, reflecting the trend of DVDs becoming popular for family-oriented entertainment at home.
""")

# Big Production Type
st.markdown("#### 3. Big Production Type")
st.markdown("""
- **Action, Adventure, and Fantasy** see notable growth during the DVD era, likely reflecting their appeal as blockbuster genres that drive high sales in physical media markets.
- **Comedy** shows a slight decline post-DVD, possibly due to the genre's shift to streaming platforms, which became more accessible post-DVD era.
""")

# Super Production Type
st.markdown("#### 4. Super Production Type")
st.markdown("""
- **Action and Adventure** maintain dominance across all eras, with **Fantasy** showing significant growth during the DVD era.
- **Science Fiction** remains relatively stable but sees a slight increase post-DVD, reflecting its appeal in the growing digital streaming market.
- **War and Western genres** remain underrepresented, likely due to limited audience demand in these genres.
""")

# General Trends
st.header("General Trends")

# General Trends Content
st.markdown("""
The data reveals that core genres like Comedy, Drama, and Action maintain their relevance across eras, although some shifts in proportions occur based on production type. 
The DVD era provided a platform for niche genres like Horror and Fantasy to thrive, likely due to their strong replay value and appeal to collectors. 
Family and Animation genres also experienced a notable boost during this period, reflecting the medium's popularity for family-friendly entertainment. 
However, the post-DVD era sees some genres, such as Comedy, decline slightly, as they transition to digital-first releases and streaming platforms.
""")

# Diversification Through DVDs
st.subheader("Diversification Through DVDs")
st.markdown("""
The accessibility of DVDs encouraged experimentation with:
- **Niche Genres**: Documentaries, indie dramas, and anime found dedicated audiences.
- **Expanded Themes**: Unique storylines thrived in home-viewing markets.
""")

# Streaming and Globalization
st.subheader("Streaming and Globalization")
st.markdown("""
The post-DVD era saw a shift toward:
- **Mainstream Genres**: Blockbusters with universal themes became the focus for theatrical releases.
- **Niche Revival Online**: Streaming platforms supported diverse genres, appealing to segmented audiences.
""")

# Different Trends per World Region
st.subheader("Different Trends per World Region")

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

countries_genres_props = get_proportions(df_countries_filtered, ['dvd_era', 'region'], 'genres')
fig = return_genre_prop_per_region(selected_regions, countries_genres_props)

st.plotly_chart(fig)


st.header("Insights and Implications")

# Revenue
st.subheader("Revenue")
st.markdown("""
- The DVD era introduced a pivotal revenue stream that benefited a variety of film genres and budgets.
- Streaming platforms have redefined revenue strategies, emphasizing global reach and licensing deals.
""")

# Budget
st.subheader("Budget")
st.markdown("""
- DVDs reduced barriers for low and mid-budget films, but their decline shifted the focus to high-budget productions.
- Streaming services continue to challenge traditional budget allocations with diverse content strategies.
""")

# Production
st.subheader("Production")
st.markdown("""
- Independent studios gained traction during the DVD era, while the post-DVD world has been shaped by streaming giants.
- The evolution of production dynamics highlights the importance of adapting to changing distribution models.
""")

# Genre
st.subheader("Genre")
st.markdown("""
- DVDs fostered genre diversity and creative experimentation, while streaming platforms have revived niche themes.
- The emphasis on algorithms and mass appeal in streaming may constrain innovation over time.
""")

st.header("Conclusion")
st.markdown("""The evolution of distribution models—from theatrical releases to DVDs to streaming—has significantly influenced the film industry’s financial strategies, production processes, and creative outputs. While the DVD era marked a golden age for revenue diversification and genre exploration, the rise of digital platforms has introduced new challenges and opportunities. As the industry continues to adapt, understanding these trends will be key to navigating the future of filmmaking.""") 


st.markdown("""
    <style>
        /* Style the Next button for both light and dark mode */
        .stButton button {
            position: fixed;
            bottom: 40px;
            right: 40px;
            background-color: rgba(0, 123, 255, 0.8); /* A nice blue with transparency */
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 20px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.06); /* Subtle shadow */
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: rgba(30, 144, 255, 1); /* Brighter blue on hover */
            transform: scale(1.05); /* Slight zoom effect */
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2), 0 2px 4px rgba(0, 0, 0, 0.1); /* Enhanced shadow */
        }

        /* Ensure compatibility with both light and dark themes */
        @media (prefers-color-scheme: light) {
            .stButton button {
                background-color: rgba(0, 123, 255, 0.9); /* Slightly darker blue for light mode */
                color: white;
            }

            .stButton button:hover {
                background-color: rgba(0, 104, 204, 1); /* Complementary hover effect */
            }
        }

        @media (prefers-color-scheme: dark) {
            .stButton button {
                background-color: rgba(30, 144, 255, 0.9); /* Brighter blue for dark mode */
                color: white;
            }

            .stButton button:hover {
                background-color: rgba(0, 104, 204, 1); /* Similar complementary effect for dark mode */
            }
        }
    </style>
""", unsafe_allow_html=True)

# Add the Next button
if st.button("Appendix →"):
    st.switch_page("appendix_page.py")


# Footer
st.markdown("---")
st.write("Created with ❤️ using Streamlit")