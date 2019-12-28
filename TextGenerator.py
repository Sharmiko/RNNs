import torch
import torch.nn as nn

class TextGenerator(nn.Module):
    
    def __init__(self, n_words, embed_size):
        
        super(TextGenerator, self).__init__()
        
        # embedding dimension
        self.embed_size = embed_size
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=n_words,
                                  embedding_dim=embed_size)
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, n_words)
        
    def forward(self, t, hidden):
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Embedding Layer
        t = self.embed(t)

        # (3) Hidden LSTM Layer
        t, hidden = self.lstm(t)
                
        # (4) Dropout Layer
        t = self.dropout(t)
        
        # (4) Fully Connected output Layer
        t = self.fc(t)
        
        return t, hidden
    
    def init_hidden(self, batch_size):
        # initialize hidden layer for first input
        return torch.zeros(100, batch_size, self.embed_size)