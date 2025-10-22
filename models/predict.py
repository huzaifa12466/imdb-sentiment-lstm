import os
import gdown
import torch
from models.model import ImdbModel
from models.utils import text_to_indices, vocab, load_glove_embeddings

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and Google Drive URLs
os.makedirs("models", exist_ok=True)
MODEL_URL = "https://drive.google.com/uc?id=1QWN7dK5YZnPKAbFi2wFHBAQsaM4-2LzA"
GLOVE_URL = "https://drive.google.com/uc?id=1EhVFs7mREZcvDuUIqX5fxLXyrSbRbw_F"
MODEL_PATH = "best_model.pth"
GLOVE_PATH = "glove.6B.100d.txt"

# Download if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(GLOVE_PATH):
    gdown.download(GLOVE_URL, GLOVE_PATH, quiet=False)

# Load embeddings and model
embedding_matrix = load_glove_embeddings(GLOVE_PATH, embedding_dim=100)

model = ImdbModel(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_size=128,
    embedding_matrix=embedding_matrix,
    freeze=False,
    dropout=0.4
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_sentiment(text: str):
    indices = text_to_indices(text)
    tensor = torch.tensor(indices).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    return {"probability": prob, "sentiment": "positive" if prob >= 0.5 else "negative"}
