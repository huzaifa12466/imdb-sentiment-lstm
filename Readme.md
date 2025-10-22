# IMDB Sentiment Analysis with LSTM

This project implements **binary sentiment analysis** on IMDB movie reviews using a **Bidirectional LSTM** network with PyTorch. The goal is to classify reviews as **positive** or **negative**.

---

## 🏗 Project Structure

```
imdb_sentiment_project/
│
├── models/
│   ├── model.py          ← LSTM model architecture
│   ├── predict.py        ← Prediction logic
│   ├── utils.py          ← Preprocessing, vocab building, embeddings
│   
│  
│
├── notebooks/
│   └── imdb_training.ipynb  ← Training & experimentation notebook
│
├── results/
│   ├── loss_acc_graph_5epoch.png
│   └── loss_acc_graph_10epoch.png
│
├── streamlit_app.py      ← Streamlit frontend for demo
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack & Libraries

* Python 3.11
* PyTorch (Bidirectional LSTM)
* NumPy
* NLTK (tokenization & stopwords)
* Streamlit (for UI demo)
* **Pretrained GloVe embeddings** for better word representation

---

## 📚 Methodology

1. **Data Preprocessing**

   * Tokenization with NLTK
   * Stopwords removal (excluding negations)
   * Vocabulary building for embeddings

2. **Model Architecture**

   * LSTM with **bidirectional layers**
   * Embedding layer initialized with **GloVe embeddings**
   * Dropout (0.4) to reduce overfitting

3. **Training Strategy**

   * Loss: Binary Cross-Entropy
   * Optimizer: Adam
   * Metrics: Accuracy, F1 Score
   * Epochs: 10

4. **Overfitting Observations**

   * Despite using dropout, overfitting occurs after 5 epochs
   * Train accuracy rises to ~99%, test accuracy stays ~87-88%
   * Using **Attention-based LSTM** or **BERT** could reduce overfitting

---

## 📈 Training Results

**Loss & Accuracy per Epoch:**

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc | F1 Score |
| ----- | ---------- | --------- | --------- | -------- | -------- |
| 1     | 0.5161     | 0.7826    | 0.6127    | 0.6882   | 0.5706   |
| 2     | 0.5038     | 0.7853    | 0.5643    | 0.6529   | 0.5248   |
| 3     | 0.3111     | 0.8743    | 0.2702    | 0.8907   | 0.8938   |
| 4     | 0.1626     | 0.9439    | 0.2720    | 0.8956   | 0.8965   |
| 5     | 0.0734     | 0.9802    | 0.3465    | 0.8888   | 0.8882   |
| 6     | 0.0386     | 0.9921    | 0.4542    | 0.8860   | 0.8871   |
| 7     | 0.0236     | 0.9962    | 0.5877    | 0.8857   | 0.8863   |
| 8     | 0.0195     | 0.9971    | 0.6071    | 0.8777   | 0.8759   |
| 9     | 0.0180     | 0.9975    | 0.6510    | 0.8762   | 0.8725   |
| 10    | 0.0143     | 0.9981    | 0.6751    | 0.8784   | 0.8791   |

💾 Final model saved (Epoch 10) | Test Acc: 0.8784

---

## 📊 Training Graphs

Graphs are available in `results/`:

* **[loss_acc_graph_5epoch.png](results/loss_acc_graph_5epoch.png)**
* **[loss_acc_graph_10epoch.png](results/loss_acc_graph_10epoch.png)**

You can **download** them directly from the `results/` folder for reference.

---

## ⚡ Usage

### Run Streamlit Demo:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Enter a review in the text box and click **Predict** to see the sentiment and probability.

### Predict via Model:

```python
from models.predict import predict_sentiment
predict_sentiment("Your review text here")
```

---

## ✨ Notes

* Pretrained **GloVe embeddings** improved generalization.
* Dropout mitigates overfitting but does not fully remove it in standard LSTM.
* Advanced models like **Attention-LSTM** or **BERT** can significantly reduce overfitting for longer training.
