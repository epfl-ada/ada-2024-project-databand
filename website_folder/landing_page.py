import streamlit as st

# This is the code to change the Streamlit settings
st.set_page_config(
    page_title="Animated Logo",
    layout="wide",
    initial_sidebar_state="collapsed",  # This hides the sidebar
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Dark mode CSS
st.markdown("""
    <style>
        /* Dark mode styles */
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
        }
        
        [data-testid="stHeader"] {
            background-color: rgba(14, 17, 23, 0);
        }
        
        .stApp {
            margin: 0;
            padding: 0;
            background-color: #0E1117;
        }
        
        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 0;
            padding-bottom: 0;
            padding-left: 0;
            padding-right: 0;
        }
        
        /* Hide hamburger menu */
        #MainMenu {
            visibility: hidden;
        }
        
        /* Hide footer */
        footer {
            visibility: hidden;
        }

        /* Style the next button */
        .stButton button {
            position: fixed;
            bottom: 40px;
            right: 40px;
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 0.5rem 2rem;
            border-radius: 20px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# code from: https://codepen.io/JavierM42/pen/wvOxaRK
# HTML code
html_code = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">

<div class="container">
  <div class="logo">
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      height="100%"
      viewBox="0 0 210 107"
      fill="currentColor"
    >
      <path d="M118.895,20.346c0,0-13.743,16.922-13.04,18.001c0.975-1.079-4.934-18.186-4.934-18.186s-1.233-3.597-5.102-15.387H81.81H47.812H22.175l-2.56,11.068h19.299h4.579c12.415,0,19.995,5.132,17.878,14.225c-2.287,9.901-13.123,14.128-24.665,14.128H32.39l5.552-24.208H18.647l-8.192,35.368h27.398c20.612,0,40.166-11.067,43.692-25.288c0.617-2.614,0.53-9.185-1.054-13.053c0-0.093-0.091-0.271-0.178-0.537c-0.087-0.093-0.178-0.722,0.178-0.814c0.172-0.092,0.525,0.271,0.525,0.358c0,0,0.179,0.456,0.351,0.813l17.44,50.315l44.404-51.216l18.761-0.092h4.579c12.424,0,20.09,5.132,17.969,14.225c-2.29,9.901-13.205,14.128-24.75,14.128h-4.405L161,19.987h-19.287l-8.198,35.368h27.398c20.611,0,40.343-11.067,43.604-25.288c3.347-14.225-11.101-25.293-31.89-25.293h-18.143h-22.727C120.923,17.823,118.895,20.346,118.895,20.346L118.895,20.346z" />
      <path d="M99.424,67.329C47.281,67.329,5,73.449,5,81.012c0,7.558,42.281,13.678,94.424,13.678c52.239,0,94.524-6.12,94.524-13.678C193.949,73.449,151.664,67.329,99.424,67.329z M96.078,85.873c-11.98,0-21.58-2.072-21.58-4.595c0-2.523,9.599-4.59,21.58-4.59c11.888,0,21.498,2.066,21.498,4.59C117.576,83.801,107.966,85.873,96.078,85.873z" />
    </svg>
  </div>
  <div class="text-container">
    <h1>The Rise and Fall of DVDs</h1>
    <p class='subtitle'>and their impact on the film industry</p>

  </div>
</div>
"""

# CSS code
css_code = """
<style>
:root {
  --logo-w: 80px;
  --logo-h: 40px;
  --duration-y: 4s;
  --duration-x: calc(var(--duration-y) * 1.5);  /* adjusted for screen aspect ratio */
}

.container {
  position: fixed;  /* changed from relative */
  top: 0;
  left: 0;
  width: 100vw;     /* changed to viewport width */
  height: 100vh;    /* changed to viewport height */
  z-index: -1;      /* puts it behind other content */
  border: none;     /* removed border */
  margin: 0;        /* removed margin */
}

.logo {
  position: absolute;
  width: var(--logo-w);
  height: var(--logo-h);
  animation:
    x var(--duration-x) linear infinite alternate,
    y var(--duration-y) linear infinite alternate,
    colorX calc(var(--duration-x) * 5) step-start infinite,
    colorY calc(var(--duration-y) * 5) step-start infinite;
  color:
    hsl(calc(
      360 / 25 * (var(--color-y) * 5 + var(--color-x)) * 1deg
    ) 100% 50%);
}

.text-container {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: rgba(14, 17, 23, 0.8);
    padding: 2rem;
    border-radius: 10px;
    border: 1px solid rgba(14, 17, 23, 0.8);
    color: white;
    opacity: 0;
    animation: fadeIn 1s ease-in forwards;
    animation-delay: 5s;
    text-align: center;
    backdrop-filter: blur(10px);
    z-index: 1;
    font-family: 'Inter', sans-serif;
}
.text-container h1 {
    font-size: 2.75rem;
    font-weight: 600;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.text-container .subtitle {
    font-size: 1.25rem;
    font-weight: 300;
    color: #e2e8f0;
    line-height: 1.6;
    margin-bottom: 1rem;
}


@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translate(-50%, -40%);
    }
    to {
        opacity: 1;
        transform: translate(-50%, -50%);
    }
}

@keyframes x {
  from { left: 0; }
  to { left: calc(100% - var(--logo-w)); }
}

@keyframes y {
  from { top: 0; }
  to { top: calc(100% - var(--logo-h)); }
}

@keyframes colorX {
  from { --color--x: 0; }
  20% { --color-x: 2; }
  40% { --color-x: 4; }
  60% { --color-x: 1; }
  80% { --color-x: 3; }
  to { --color-x: 0; }
}

@keyframes colorY {
  from { --color-y: 0; }
  20% { --color-y: 2; }
  40% { --color-y: 4; }
  60% { --color-y: 1; }
  80% { --color-y: 3; }
  to { --color-y: 0; }
}
</style>
"""

# Display the HTML and CSS
st.components.v1.html(css_code + html_code, height=700)

# Add the Next button
if st.button("Next →"):
    st.switch_page("pages/Matt_damon.py")

st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 40px; color: white; text-align: center; font-size: 14px;">
        Or you can stay here and watch if the DVD lands on a corner
    </div>
    """,
    unsafe_allow_html=True
)



