import streamlit as st
from models.predict import predict_sentiment

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review below to see whether it's **positive** or **negative**.")

review = st.text_area("Movie Review", height=150)

if st.button("Predict Sentiment"):
    if review.strip():
        result = predict_sentiment(review)
        st.markdown(f"### Sentiment: **{result['sentiment'].capitalize()}**")
        st.markdown(f"**Probability:** {result['probability']:.2f}")
    else:
        st.warning("Please enter a review.")
