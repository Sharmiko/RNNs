from TextGenerator import TextGenerator
import data

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 100
BATCH_SIZE = data.BATCH_SIZE
EPOCHS = 1000

# Create text generator model
model = TextGenerator(data.n_words, EMBED_DIM)
model.to(device)

# create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# define text loader 
textloader = data.text_loader

# initalize first hidden state
hidden = model.init_hidden(BATCH_SIZE).to(device)

# Training loop
for epoch in tqdm(range(EPOCHS)):
    
    for x, y in textloader:
        
        # convert numpy arrays into torch tensors
        x = torch.Tensor(x).long().to(device)
        y = torch.Tensor(y).long().to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        output, hidden = model(x, hidden)
        
        # calculate loss
        loss = criterion(output.transpose(1, 2), y)
        
        loss.backward()
        optimizer.step()
        
    print("Epoch {}/{}; Loss -> {}".format(epoch + 1, EPOCHS, loss))
        


def generate():
    
    words = ["fox", "am"]
    
    hidden = model.init_hidden(BATCH_SIZE)
    for w in words:
        ix = torch.tensor([[data.word2idx[w]]]).to(device)
        output, hidden = model(ix, hidden)
        
    _, top_ix = torch.topk(output[0], k=5)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    
    words.append(data.idx2word[choice])
    
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, hidden = model(ix, hidden)
        
        _, top_ix = torch.topk(output[0], k=5)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(data.idx2word[choice])
        
    print(' '.join(words).encode('utf-8'))
    
generate()

    
    