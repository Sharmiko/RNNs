import torch
import torch.nn as nn

class MusicRNN(nn.Module):
    
    def __init__(self, n_notes, n_durations, embed_size):
        
        super(MusicRNN, self).__init__()
        
        self.embed_notes = nn.Embedding(n_notes, embed_size)
        self.embde_dur = nn.Embedding(n_durations, embed_size)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm1 = nn.LSTM(200, 256)
        self.lstm2 = nn.LSTM(256, 256)
        
        self.linear = nn.Linear(256, 1)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        self.out_notes = nn.Linear(256, n_notes)
        self.out_durations = nn.Linear(256, n_durations)
        
    def forward(self, notes, durations):
        
        # (1) Input Layer: Notes and Durations
        notes = notes
        durations = durations

        # (2) Embedding Layers: Notes and Durations
        notes = self.embed_notes(notes)
        durations = self.embde_dur(durations)
        
        print(notes.size(), durations.size())
        # (3) Concatenation Layer: Notes + Durations
        t = torch.cat([durations, notes], axis=2)
        
        # (4) First LSTM Layer
        t, hidden = self.lstm1(t)
        t = self.dropout(t)

        # (5) Second LSTM Layer
        t_x, hidden_x = self.lstm2(t)
        t = self.dropout(t_x)

        # (6) Fully Connected Layer
        t = self.linear(t)

        # (7) Tanh Activation Function
        t = self.tanh(t)

        # (8) Repeat Tensor 256 Times
        t = t.repeat(1, 1, 256)

        # (9) Multiply Tensor and output from second LSTM Layer
        t = torch.mul(t, t_x)
        
        # (10) Sum Values along axis=1
        t = torch.sum(t, axis=1)
        
        # (11) Output Layer for notes and durations
        out_notes = self.out_notes(t)
        out_durations = self.out_durations(t)
        
        return self.softmax(out_notes), self.softmax(out_durations)