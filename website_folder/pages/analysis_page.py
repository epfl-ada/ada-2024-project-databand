import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="The Fall of DVD: Analysis with TMDB Data", layout="wide")

# Sidebar 
st.sidebar.title("Navigation")
sections = [
    "Introduction",
    "Revenue Analysis",
    "Budget Analysis",
    "Production Analysis",
    "Genres Analysis",
    "Conclusion"
]
selection = st.sidebar.radio("Go to:", sections)

# Helper function to scrollable sections
def section_divider(title):
    st.markdown(f"""<div style='margin-top: 50px; margin-bottom: 20px; border-top: 2px solid #ccc;'>
                <h2 style='margin-top: 20px;'>{title}</h2></div>""", unsafe_allow_html=True)

# Header Section
if selection == "Introduction" or st.session_state.get('scroll', True):
    st.title("The Fall of DVD: Analysis with TMDB Data")
    st.markdown(
        """### Exploring the Impact of DVDs on the Film Industry
        This website presents an in-depth analysis of the rise and fall of DVDs and their impact on movie revenue, budgets, production companies, and genres using data from TMDB.

        Below, you'll find answers to research questions structured into four key areas: Revenue, Budget, Production, and Genres. Navigate through the sections to explore the findings.
        """
    )
    section_divider("Revenue Analysis")

# Revenue Section
if selection == "Revenue Analysis" or st.session_state.get('scroll', True):
    st.title("Revenue Analysis")

    st.subheader("Impact of DVDs on Overall Movie Revenue")
    st.markdown("Analyzing how DVD sales contributed to overall movie revenue.")

    # Example chart: Revenue trends
    revenue_data = pd.DataFrame({"Year": [2000, 2005, 2010, 2015, 2020], "Revenue": [50, 80, 100, 90, 60]})
    fig = px.line(revenue_data, x="Year", y="Revenue", title="Revenue Trends over the Years", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue by Categories")
    st.markdown("Examining revenue shifts for different genres and budgets over time.")
    
    # Placeholder for detailed analysis or interactive visualizations
    st.info("Detailed findings and interactive visualizations coming soon!")
    section_divider("Budget Analysis")

# Budget Section
if selection == "Budget Analysis" or st.session_state.get('scroll', True):
    st.title("Budget Analysis")

    st.subheader("Did DVDs Level the Playing Field for Low-Budget Movies?")
    st.markdown("Investigating how DVDs influenced small production studios.")

    # Example visualization: Budget categories
    budget_data = pd.DataFrame({"Category": ["Low", "Medium", "High"], "Revenue Impact": [10, 20, 40]})
    fig = px.bar(budget_data, x="Category", y="Revenue Impact", title="Impact of DVDs on Different Budget Categories")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact of DVD Decline on Low- and Mid-Budget Movies")
    st.markdown("Assessing the changes in the market dynamics post-DVD era.")
    
    st.info("Detailed findings and interactive visualizations coming soon!")
    section_divider("Production Analysis")

# Production Section
if selection == "Production Analysis" or st.session_state.get('scroll', True):
    st.title("Production Analysis")

    st.subheader("Change in Dominant Production Companies")
    st.markdown("Examining the shifts in production company dominance across eras.")
    
    # Example table
    production_data = pd.DataFrame({"Era": ["Pre-DVD", "DVD Era", "Post-DVD"], "Top Company": ["Company A", "Company B", "Company C"]})
    st.table(production_data)

    st.subheader("Emergence of New Players Post-DVD Era")
    st.markdown("Exploring the rise of new production companies after the DVD era.")
    
    st.info("Detailed findings and interactive visualizations coming soon!")
    section_divider("Genres Analysis")

# Genres Section
if selection == "Genres Analysis" or st.session_state.get('scroll', True):
    st.title("Genres Analysis")

    st.subheader("How DVDs Influenced the Rise of New Genres")
    st.markdown("Looking at genre diversity during the DVD era.")

    # Example visualization: Genres comparison
    genre_data = pd.DataFrame({"Era": ["Pre-DVD", "DVD Era", "Post-DVD"], "Genre Diversity": [5, 8, 6]})
    fig = px.bar(genre_data, x="Era", y="Genre Diversity", title="Genre Diversity Across Eras")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact on Themes within Genres")
    st.markdown("Exploring shifts in major themes within movie genres.")
    
    st.info("Detailed findings and interactive visualizations coming soon!")
    section_divider("Conclusion")

# Conclusion Section
if selection == "Conclusion" or st.session_state.get('scroll', True):
    st.title("Conclusion")
    st.markdown(
        """### Summary of Findings
        - DVDs had a noticeable impact on revenue, especially for specific genres and budget categories.
        - The decline of DVDs led to shifts in production dynamics, favoring high-budget movies.
        - Genre diversity peaked during the DVD era and evolved with streaming services.
        
        Explore the visualizations and insights shared in each section to dive deeper into the findings!
        """
    )
    # st.image("summary_image.jpg", use_column_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")
