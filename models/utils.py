import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'never'}
stop_words = stop_words - negations

vocab = {'<UNK>': 0}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens).strip()

def build_vocab(sentence):
    for word in word_tokenize(sentence):
        if word not in vocab:
            vocab[word] = len(vocab)

def text_to_indices(text):
    indexed_text = []
    for word in word_tokenize(text):
        indexed_text.append(vocab.get(word, vocab['<UNK>']))
    return indexed_text

def load_glove_embeddings(glove_path, embedding_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            if len(values) == embedding_dim + 1:
                word = values[0]
                if word in vocab:
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[vocab[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float)
