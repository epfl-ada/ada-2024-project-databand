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

# Section 2: Next Content
st.markdown(
    """
    <div id="section2" style="margin-top: 100vh; padding-top: 50px;">
        <h2 style="text-align: center; font-family: 'Inter', sans-serif; color: #343a40;">Next Section Content</h2>
        <p style="font-size: 1.2rem; line-height: 1.6; color: #495057; text-align: center;">
            This is where the content of the next section would go. You can add graphs, images, or any additional details here.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown("---")
st.write("Created with ❤️ using Streamlit")
