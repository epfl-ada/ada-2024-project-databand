import streamlit as st
import streamlit.components.v1 as components
import os
from pathlib import Path

current_dir = Path(__file__).parent

file_path = current_dir / "1_Analysis.py"
print(f"File exists: {file_path.exists()}")
print(f"File is file: {file_path.is_file()}")

# Page Configurations
st.set_page_config(
    page_title="Matt Damon Explains",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styles
st.markdown(
    """
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
st.title("Matt Damon Explains")

# Section 1: Main Content
st.markdown(
    """
    <div class="content-container">
        <div class="explanation hover-box">
            <h3>Basically...</h3>
            <p>
            The appearance of DVDs in the 1990s had a resounding impact on the film industry. It provided studios with a whole new revenue stream and provided wider accessibility to movies. More importantly, the popularity of DVDs meant that even if a movie did not perform well at the box office, studios could still generate a profit through sales of DVDs. Throughout the 2000s, however, the industry shifted away from physical media to a digital streaming industry, pushing studios to rely on successful theatrical releases notably, blockbusters. In this interview, Matt Damon explains his experience with this problem and states that today, most of the movies he starred in could not be made.
            </p>
        </div>
        <div class="video-container hover-box">
            <h2>Watch the Video</h2>
            <div>
                <iframe width="100%" height="315" src="https://www.youtube.com/embed/Jx8F5Imd8A8" frameborder="0" allowfullscreen></iframe>
            </div>
        </div>
    </div>
    <style>
        .hover-box {
            background: #000000;
            color: #ffffff;
            border: 2px solid #ffffff;
            transition: border-color 0.3s ease;
        }
        .hover-box:hover {
            border-color: #ff5722; /* Orange border on hover */
        }
        .hover-box h3, .hover-box h2 {
            color: #ffffff;
        }
        .hover-box p {
            font-size: 1.2rem;
            line-height: 1.6;
            color: #ffffff;
        }
    </style>
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
        /* Style the Back button for both light and dark mode */
        .stButton button {
            position: fixed;
            bottom: 40px;
            right: 40px; /* Position at bottom-right */
            background-color: white; /* White background */
            color: rgba(255, 87, 34, 0.8); /* Orange text */
            border: 2px solid rgba(255, 87, 34, 0.8); /* Orange border */
            padding: 0.5rem 2rem;
            border-radius: 20px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.06); /* Subtle shadow */
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #f0f0f0; /* Greyish background on hover */
            color: rgba(255, 87, 34, 1); /* Orange text on hover */
            border: 2px solid rgba(255, 69, 0, 1); /* Brighter orange border on hover */
            transform: scale(1.05); /* Slight zoom effect */
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2), 0 2px 4px rgba(0, 0, 0, 0.1); /* Enhanced shadow */
        }

        /* Ensure compatibility with both light and dark themes */
        @media (prefers-color-scheme: light) {
            .stButton button {
                background-color: white; /* White background in light mode */
                color: rgba(255, 87, 34, 0.8); /* Orange text */
                border: 2px solid rgba(255, 87, 34, 0.8); /* Orange border */
            }

            .stButton button:hover {
                background-color: #f0f0f0; /* Greyish background */
                border: 2px solid rgba(255, 69, 0, 1); /* Brighter orange border */
            }
        }

        @media (prefers-color-scheme: dark) {
            .stButton button {
                background-color: white; /* White background in dark mode */
                color: rgba(255, 87, 34, 0.8); /* Orange text */
                border: 2px solid rgba(255, 87, 34, 0.8); /* Orange border */
            }

            .stButton button:hover {
                background-color: #f0f0f0; /* Greyish background */
                border: 2px solid rgba(255, 69, 0, 1); /* Brighter orange border */
            }
        }
    </style>
""", unsafe_allow_html=True)

# Add the Next button
if st.button("Let's investigate →"):
    st.switch_page("pages/02_Analysis.py")

# Footer
st.markdown("---")
st.write("Created with ❤️ using Streamlit")
