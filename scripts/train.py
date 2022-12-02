# train.py
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import os

os.chdir(os.path.dirname(__file__))

# Loading configuration
configuration = {}
with open("config.json", "r") as config_file:
    configuration = json.loads(config_file.read())

with open(configuration.get("dataset"), 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = configuration.get("num_epochs")
batch_size = configuration.get("batch_size")
learning_rate = configuration.get("learning_rate")
input_size = len(X_train[0])
hidden_size = configuration.get("hidden_size")
output_size = len(tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    print("Training with the following parameters:")
    print("Epochs: " + str(num_epochs))
    print("Batch size: " + str(batch_size))
    print("Input size: " + str(input_size))
    print("Hidden size: " + str(hidden_size))
    print("Output size: " + str(output_size))
    print("Learning rate: " + str(learning_rate))
    print("Early stop: " + str(configuration.get("early_stop")))
        
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Working on: " + str(device))

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    for epoch in range(num_epochs):
        #print(str(epoch) + "/" + str(num_epochs) + " epoch training...")
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#        if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() < 0.05:
            if(configuration.get("early_stop")):
                break


    print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }
    
    # TODO Somehow save the model to continue training
    # LINK https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    FILE = configuration.get("model")
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')
