import torch
import numpy as np
from collections import Counter

SEQUENCE_LENGTH = 20
BATCH_SIZE = 32

filepath = "Aesop'sFables.txt"

def getData(pathToFile):
    
    # open file
    with open(pathToFile, encoding='utf-8-sig') as f:
        text = f.read()
    text = text.split()
    
    # count unique words
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create index to word and vice-versa dictionaries
    idx2word = {k: w for k, w in enumerate(sorted_vocab)}
    word2idx = {w: k for k, w in idx2word.items()}
    # number of entries in dictionary
    n_words = len(idx2word)
    
    # convert text to integer values
    text2int = [word2idx[w] for w in text]
    #calculate number of batches
    num_batches = int(len(text2int) / (SEQUENCE_LENGTH * BATCH_SIZE))
    
    # create input text 
    text_X = text2int[:num_batches * BATCH_SIZE * SEQUENCE_LENGTH]
    # create output text
    text_y = np.zeros_like(text_X)
    
    text_y[:-1] = text_X[1:]
    text_y[-1] = text_X[0]
    
    text_X = np.reshape(text_X, (BATCH_SIZE, -1))
    text_y = np.reshape(text_y, (BATCH_SIZE, -1))
    
    return idx2word, word2idx, n_words, text_X, text_y


def getBatches(X, y):
    # create generator object that yields batches of X and y
    num_batches = np.prod(X.shape) // (SEQUENCE_LENGTH * BATCH_SIZE)
    for i in range(0, num_batches * SEQUENCE_LENGTH, SEQUENCE_LENGTH):
        yield X[:, i:i + SEQUENCE_LENGTH], y[:, i:i + SEQUENCE_LENGTH]
        

idx2word, word2idx, n_words, text_X, text_y = getData(filepath)

text_loader = getBatches(text_X, text_y)