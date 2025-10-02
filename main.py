# -------------------------------
# IMDb Sentiment Analysis App
# TF-IDF + ML (Streamlit)
# -------------------------------

import streamlit as st
import joblib
import re
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- Page Configuration ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a next-level UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Poppins:wght@300;400&display=swap');

    /* General Styles */
    body {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #000000;
        color: #f5f5f5;
        overflow: hidden; /* To contain the vanta background */
    }
    
    /* Vanta.js Canvas */
    #vanta-canvas {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        top: 0;
        left: 0;
    }

    /* Main containers for a 3D tilting "glassmorphism" effect */
    .main-container {
        background: rgba(10, 10, 25, 0.5);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.3), 0 0 40px rgba(255, 0, 255, 0.3);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 2px solid transparent;
        border-image: linear-gradient(45deg, #FF00FF, #00BFFF);
        border-image-slice: 1;
        animation: rotateBorder 6s linear infinite;
        margin-bottom: 2rem;
        transition: box-shadow 0.4s ease-in-out, transform 0.4s ease;
        transform-style: preserve-3d;
    }

    .main-container:hover {
        transform: perspective(1000px) rotateY(5deg) scale(1.02);
        box-shadow: 0 0 30px rgba(0, 191, 255, 0.5), 0 0 60px rgba(255, 0, 255, 0.5);
    }
    
    @keyframes rotateBorder {
      0% { border-image-source: linear-gradient(45deg, #FF00FF, #00BFFF); }
      50% { border-image-source: linear-gradient(225deg, #FF00FF, #00BFFF); }
      100% { border-image-source: linear-gradient(405deg, #FF00FF, #00BFFF); }
    }
    
    /* Glowing result boxes */
    .result-box.positive-glow {
        box-shadow: 0 0 40px rgba(40, 167, 69, 1), 0 0 20px rgba(40, 167, 69, 1) inset;
        border-image: linear-gradient(45deg, #28a745, #98FB98, #28a745);
        border-image-slice: 1;
    }
    .result-box.negative-glow {
        box-shadow: 0 0 40px rgba(220, 53, 69, 1), 0 0 20px rgba(220, 53, 69, 1) inset;
        border-image: linear-gradient(45deg, #dc3545, #FFB6C1, #dc3545);
        border-image-slice: 1;
    }

    /* Decorated Text Styling with Animated Gradient */
    h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        text-align: center;
        font-weight: 700;
        background: linear-gradient(90deg, #FF00FF, #00BFFF, #FF00FF);
        background-size: 200% auto;
        color: #fff;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 5s linear infinite;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    @keyframes textShine {
      to { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #f5f5f5;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 3rem;
        animation: fadeInDown 1.5s;
    }
    h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: #00BFFF;
        text-shadow: 0 0 8px #00BFFF;
    }

    .stTextArea textarea {
        background-color: rgba(0, 0, 0, 0.5);
        color: #f5f5f5;
        border-radius: 10px;
        border: 2px solid #00BFFF;
        font-size: 1.1rem;
        transition: box-shadow 0.3s, border-color 0.3s;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 20px #00BFFF;
        border-color: #FF00FF;
    }

    /* Aurora Button */
    .stButton>button {
        position: relative;
        color: #ffffff;
        background: transparent;
        border-radius: 10px;
        border: 2px solid #00BFFF;
        padding: 14px 32px;
        font-size: 18px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px 0 rgba(0, 191, 255, 0.7);
        width: 100%;
        overflow: hidden;
    }
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, transparent, rgba(0, 191, 255, 0.6), transparent);
        transition: left 0.7s;
    }
    .stButton>button:hover:before {
        left: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 30px 0 rgba(0, 191, 255, 1);
        transform: translateY(-4px);
        background: #00BFFF;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(0, 4, 40, 0.7);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
    }

    /* Result styling */
    .result-box { text-align: center; padding: 2.5rem; }
    .sentiment-icon { font-size: 7rem; animation: popIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55); text-shadow: 0 0 25px; }
    .sentiment-text {
        font-size: 3.5rem;
        font-weight: 700;
        text-transform: uppercase;
        animation: fadeIn 1.2s ease-in-out;
    }
    .positive { color: #28a745; text-shadow: 0 0 20px #28a745; }
    .negative { color: #dc3545; text-shadow: 0 0 20px #dc3545; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #00BFFF , #FF00FF); }
    .stCodeBlock { background-color: rgba(0,0,0,0.4) !important; border: 1px solid #00BFFF; }

    .review-quote {
        font-style: italic;
        color: #ccc;
        border-left: 4px solid #00BFFF;
        padding-left: 1.5rem;
        margin-top: 2rem;
        text-align: left;
    }
    .footer { text-align: center; color: #aaa; font-size: 0.9rem; font-weight: 300; }

    /* Animations */
    @keyframes fadeIn { 0% { opacity: 0; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }
    @keyframes fadeInDown { 0% { opacity: 0; transform: translateY(-30px); } 100% { opacity: 1; transform: translateY(0); } }
    @keyframes popIn { 0% { transform: scale(0); } 100% { transform: scale(1); } }
</style>
""", unsafe_allow_html=True)

# --- Vanta.js 3D Background ---
st.html("""
<div id="vanta-canvas"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js"></script>
<script>
VANTA.GLOBE({
  el: "#vanta-canvas",
  mouseControls: true,
  touchControls: true,
  gyroControls: false,
  minHeight: 200.00,
  minWidth: 200.00,
  scale: 1.00,
  scaleMobile: 1.00,
  color: 0xff00ff,
  color2: 0x00ffff,
  backgroundColor: 0x0,
  size: 0.8
})
</script>
""")


# --- NLTK Resource Downloads ---
def download_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        with st.spinner("Downloading NLTK 'stopwords' resource..."):
            nltk.download('stopwords')
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        with st.spinner("Downloading NLTK 'wordnet' resource..."):
            nltk.download('wordnet')

download_nltk_resources()


# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model_path = "sentiment_model.pkl"
    tfidf_path = "tfidf_vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        st.error(f"Error: Model or TF-IDF file not found. Please ensure the paths are correct.")
        return None, None
    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf
    except Exception as e:
        st.error(f"An error occurred while loading the files: {e}")
        return None, None

model, tfidf = load_model_and_vectorizer()

# --- Text Preprocessing Function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Sidebar ---
with st.sidebar:
    st.header("🎬 About the App")
    st.write("""
        This application analyzes the sentiment of movie reviews from IMDb. 
        Using a machine learning model, it predicts whether a given review is **Positive** or **Negative**.
    """)
    st.header("⚙️ How It Works")
    st.write("""
        1.  **Input:** You provide a movie review.
        2.  **Preprocessing:** The text is cleaned and prepared.
        3.  **TF-IDF:** The text is converted into numerical data.
        4.  **Prediction:** A pre-trained model predicts the sentiment and its confidence.
    """)
    st.info("The model and vectorizer are cached for faster performance.")

# --- Main Page Layout ---
st.title("Sentiment Analyzer")
st.markdown("<p class='subtitle'>Harnessing AI to Decode the Emotion in Movie Reviews</p>", unsafe_allow_html=True)

# Main container for the input
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("Enter Your Movie Review")
    user_input = st.text_area(
        "Label:", 
        height=200,
        placeholder="e.g., 'This film was a cinematic triumph! The performances were breathtaking and the story was deeply moving.'",
        label_visibility="collapsed"
    )

    if st.button("Analyze Sentiment"):
        if model and tfidf:
            if user_input.strip():
                with st.spinner('Analyzing...'):
                    clean_input = preprocess(user_input)
                    input_tfidf = tfidf.transform([clean_input])
                    prediction = model.predict(input_tfidf)[0]
                    prediction_proba = model.predict_proba(input_tfidf)
                    
                    st.session_state['prediction'] = prediction
                    st.session_state['probability'] = prediction_proba
                    st.session_state['review'] = user_input
            else:
                st.warning("⚠️ Please enter a review to analyze.")
        else:
            st.warning("The model is not loaded. Please check the console for errors.")
    st.markdown('</div>', unsafe_allow_html=True)

# Display results if they exist in the session state
if 'prediction' in st.session_state:
    with st.container():
        prediction = st.session_state['prediction']
        probability = st.session_state['probability']
        
        # Assuming model.classes_ is ['negative', 'positive']
        neg_prob = probability[0][0]
        pos_prob = probability[0][1]
        
        glow_class = ""
        if str(prediction).lower() == 'positive':
            glow_class = "positive-glow"
            confidence = pos_prob
        else:
            glow_class = "negative-glow"
            confidence = neg_prob

        st.markdown(f'<div class="main-container result-box {glow_class}">', unsafe_allow_html=True)
        
        if str(prediction).lower() == 'positive':
            st.markdown('<p class="sentiment-icon">👍</p>', unsafe_allow_html=True)
            st.markdown('<p class="sentiment-text positive">Positive</p>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<p class="sentiment-icon">👎</p>', unsafe_allow_html=True)
            st.markdown('<p class="sentiment-text negative">Negative</p>', unsafe_allow_html=True)
            
        st.subheader(f"Confidence: {confidence:.0%}")
        st.progress(confidence)
        
        st.markdown(f"<p class='review-quote'>Your review: <em>'{st.session_state['review']}'</em></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Expander for example reviews
with st.expander("Show Example Reviews"):
    st.markdown("#### Positive Example")
    st.code("An absolute masterpiece. The acting, cinematography, and storyline were all perfect. A must-see!")

    st.markdown("#### Negative Example")
    st.code("A complete waste of time. The plot was predictable and the characters were incredibly dull.")

# --- Footer ---
st.write("---")
st.markdown("<p class='footer'>Powered by <strong>Streamlit</strong>, <strong>TF-IDF</strong>, and a <strong>Machine Learning Model</strong>.</p>", unsafe_allow_html=True)
