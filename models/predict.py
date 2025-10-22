import os
import json
import gdown
import torch
from models.model import ImdbModel
from models.utils import text_to_indices, load_glove_embeddings

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Google Drive URLs and local paths
# ---------------------------
MODEL_URL = "https://drive.google.com/uc?id=1QWN7dK5YZnPKAbFi2wFHBAQsaM4-2LzA"
GLOVE_URL = "https://drive.google.com/uc?id=1EhVFs7mREZcvDuUIqX5fxLXyrSbRbw_F"
VOCAB_URL = "https://drive.google.com/uc?id=1Sl7wvB3qFSMemPh-Z5pGfjjPX3mVodDJ"

MODEL_PATH = "best_model.pth"
GLOVE_PATH = "glove.6B.100d.txt"
VOCAB_PATH = "vocab.json"

# ---------------------------
# Helper function to download files if missing
# ---------------------------
def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(url, path, quiet=False)

# ---------------------------
# Load vocab
# ---------------------------
def load_vocab(vocab_path=VOCAB_PATH, vocab_url=VOCAB_URL):
    download_if_missing(vocab_url, vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if '<UNK>' not in vocab:
        vocab['<UNK>'] = 0
    return vocab

# ---------------------------
# Load embeddings
# ---------------------------
def load_embeddings(glove_path=GLOVE_PATH, vocab=None):
    download_if_missing(GLOVE_URL, glove_path)
    return load_glove_embeddings(glove_path, embedding_dim=100, vocab=vocab)

# ---------------------------
# Load model
# ---------------------------
def load_model(model_path=MODEL_PATH, vocab=None, embedding_matrix=None, device=device):
    download_if_missing(MODEL_URL, model_path)
    model = ImdbModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_size=128,
        embedding_matrix=embedding_matrix,
        freeze=False,
        dropout=0.4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ---------------------------
# Full initialization for Streamlit / scripts
# ---------------------------
def initialize_model():
    vocab = load_vocab()
    embedding_matrix = load_embeddings(vocab=vocab)
    model = load_model(vocab=vocab, embedding_matrix=embedding_matrix)
    return model, vocab

# ---------------------------
# Prediction function
# ---------------------------
def predict_sentiment(text: str, model, vocab):
    indices = text_to_indices(text, vocab)
    tensor = torch.tensor(indices).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    sentiment = "positive" if prob >= 0.5 else "negative"
    return {"probability": prob, "sentiment": sentiment}

