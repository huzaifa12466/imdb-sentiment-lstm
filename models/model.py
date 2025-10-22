import torch
import torch.nn as nn

class ImdbModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, embedding_matrix=None, freeze=False, dropout=0.4):
        super(ImdbModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(1)

def load_model(model_path, vocab_size, embed_dim=100, hidden_size=128, embedding_matrix=None, device='cpu'):
    model = ImdbModel(vocab_size, embed_dim, hidden_size, embedding_matrix)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
