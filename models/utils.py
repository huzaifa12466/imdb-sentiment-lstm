import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import numpy as np

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never'}
stop_words = stop_words - negations

# ---------------------------
# Text processing functions
# ---------------------------
def clean_text(text: str) -> str:
    """
    Lowercase, remove HTML tags, URLs, non-alphabetic characters, stopwords (except negations)
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens).strip()

def text_to_indices(text: str, vocab: dict) -> list[int]:
    """
    Convert a string to a list of indices based on vocab
    """
    indexed_text = []
    for word in word_tokenize(text):
        indexed_text.append(vocab.get(word, vocab.get('<UNK>', 0)))
    if not indexed_text:
        indexed_text = [vocab.get('<UNK>', 0)]
    return indexed_text

def load_glove_embeddings(glove_path: str, embedding_dim=100, vocab=None) -> torch.Tensor:
    """
    Load GloVe embeddings. Words not found in GloVe get random vectors.
    """
    if vocab is None:
        raise ValueError("vocab must be provided to load embeddings")
    
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim)).astype(np.float32)
    
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != embedding_dim + 1:
                continue
            word = values[0]
            if word in vocab:
                vector = np.array(values[1:], dtype=np.float32)
                embeddings[vocab[word]] = vector
    
    return torch.tensor(embeddings, dtype=torch.float32)

