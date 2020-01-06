import torch
import torch.nn as nn
import torch.optim as optim

from MusicData import MusicData
from MusicRNN import MusicRNN

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Get data
music_data = MusicData('/home/sharmi/Documents/ART-AI/RNNs/data/cello', False)
# Create data loaders
pitch_input, duration_input, pitch_output, duration_output = music_data.data_loader(batch_size=4)

# create model
model = MusicRNN(music_data.n_notes, music_data.n_durations, embed_size=100)
model.to(device)

# create optimizer and loss function
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


EPOCHS = 2000000

# zip data loaders
ziped_loader = zip(pitch_input, duration_input, pitch_output, duration_output)

for epoch in tqdm(range(EPOCHS)):

    for pitch_input, duration_input, pitch_output, duration_output in ziped_loader:

        # convert tensors to model
        pitch_input = pitch_input.to(device)
        duration_input = duration_input.to(device)
        pitch_output = pitch_output.to(device)
        duration_output = duration_output.to(device)

        # zero gradients
        optimizer.zero_grad()

        out_notes, out_durations = model(pitch_input, duration_input)

        # calculate loss

        loss_notes = criterion(out_notes, pitch_output)
        loss_duration = criterion(out_durations, duration_output)

        loss = (loss_notes + loss_duration) / 2
        loss.backward()

        optimizer.step()

    print("Epoch {} / {}; Loss -> {}".format(epoch + 1, EPOCHS, loss.item()))
