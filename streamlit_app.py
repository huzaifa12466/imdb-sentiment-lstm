import streamlit as st
from models.predict import initialize_model, predict_sentiment

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis", 
    page_icon="🎬", 
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>🎬 IMDB Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a movie review below to see whether it's <b>positive</b> or <b>negative</b>.</p>", unsafe_allow_html=True)

# ---------------------------
# Initialize model (runs once)
# ---------------------------
@st.cache_resource
def load_model_once():
    model, vocab = initialize_model()
    return model, vocab

with st.spinner("Loading model..."):
    model, vocab = load_model_once()

# ---------------------------
# Input area
# ---------------------------
with st.container():
    review = st.text_area("Movie Review", height=180, placeholder="Type your movie review here...")

    if st.button("Predict Sentiment"):
        if review.strip():
            result = predict_sentiment(review, model=model, vocab=vocab)
            
            # Colored sentiment badge
            sentiment_color = "green" if result['sentiment'] == "positive" else "red"
            st.markdown(
                f"<h2 style='text-align:center; color:{sentiment_color};'>Sentiment: {result['sentiment'].capitalize()}</h2>", 
                unsafe_allow_html=True
            )
            
            # Probability progress bar
            st.progress(result['probability'])
            st.markdown(f"<p style='text-align:center;'>Probability: {result['probability']:.2f}</p>", unsafe_allow_html=True)
            
        else:
            st.warning("Please enter a review.")
