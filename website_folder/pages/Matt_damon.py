import streamlit as st
import streamlit.components.v1 as components

# Page Configurations
st.set_page_config(
    page_title="YouTube Video with Explanation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styles
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    .main { background-color: #f8f9fa; }
    h1 {
        color: #343a40;
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .content-container {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 2rem;
        padding: 2rem;
    }
    .explanation {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: 'Inter', sans-serif;
        flex: 1;
        max-height: 600px;
        overflow-y: auto;
    }
    .video-container {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("Explore and Understand: Video and Explanation")

# Section 1: Main Content
st.markdown(
    """
    <div class="content-container">
        <div class="explanation">
            <h2>Explanation of the Video</h2>
            <p style="font-size: 1.2rem; line-height: 1.6; color: #495057;">
            The emergence of DVDs in the 1990s had a resounding impact on the film industry, providing a wider accessibility to movies and a new revenue stream. The popularity of DVDs meant that even if a movie did not perform well at the box office, studios could still generate revenue through sales of DVDs. Throughout the 2000s, however, the industry shifted away from physical media to a digital streaming industry, pushing studios to rely on successful theatrical releases notably, blockbusters. Matt Damon was the first to shine a light on this problem. In an interview, he explains that most movies that he starred in could no longer be made today due to a reliance on revenue from the box office.
            </p>
        </div>
        <div class="video-container">
            <h2>Watch the Video</h2>
            <div>
                <iframe width="100%" height="315" src="https://www.youtube.com/embed/Jx8F5Imd8A8" frameborder="0" allowfullscreen></iframe>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# JavaScript for Smooth Scrolling
components.html(
    """
    <script>
        window.addEventListener('scroll', function() {
            const scrollPosition = window.scrollY + window.innerHeight;
            const pageHeight = document.body.scrollHeight;
            if (scrollPosition >= pageHeight - 10) {
                const nextSection = document.getElementById('section2');
                if (nextSection) {
                    nextSection.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    </script>
    """,
    height=0,
)


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
if st.button("Next →"):
    st.switch_page("pages/analysis_page.py")

# Footer
st.markdown("---")
st.write("Created with ❤️ using Streamlit")
