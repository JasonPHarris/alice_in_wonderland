# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:23:47 2022

@author: Jason Harris
This code is to use Keras and tensorflow to learn and generate predictive text. 
I utilized the tutorials at the following websites to build this code:
    https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/
    https://www.renom.jp/notebooks/tutorial/time_series/text_generation_using_character_level_language_model/notebook.html
    https://blog.jaysinha.me/train-your-first-lstm-model-for-text-generation/
    http://ling.snu.ac.kr/class/cl_under1801/DL-LSTMTextGenAlice.pdf
    https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
    https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys
import os                                       # importing os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # setting tensorflow compiler flags

filename = "alice_in_wonderland.txt"            # loading the ascii text file
raw_text = open(filename).read()
raw_text = raw_text.lower()                     # converting the text to all lower case

chars = sorted(list(set(raw_text)))             # creating a sorted list of unique characters in the text file
char_to_int = dict((c, i) for i, c in enumerate(chars)) # assigning numerical values to the unique cahracters

n_chars = len(raw_text)                         # generating the number of total characters in the file
n_vocab = len(chars)                            # generating the number of unique characters in the file
print("Total Characters: " + str(n_chars))      # printing the total
print("Total unique characters: " +str(n_vocab))# printing the total of unique characters

seq_length = 100                                # setting the length of the dataset input and output pairs
dataX= []
dataY = []
for i in range (0, n_chars - seq_length, 1):    # starting the loop to build the sequence
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)           # printing the number of sequence patterns

X = np.reshape(dataX, (n_patterns, seq_length, 1)) # reshaping X to conform to [samples, time steps, features]
X = X/float(n_vocab)                            # normalizing X
y = np_utils.to_categorical(dataY)

model = Sequential()                            # define the LSTM model
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),return_sequences=True)) # small LSTM's generally make more errors, in less time, Larger LSTM's learn more and create less errors
model.add(Dropout(0.2))
model.add(LSTM(256))                                                             # secondary LSTM taking input from the first LSTM Implementing this LSTM officiall makes this a RNN
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}--bigger.hdf5" # defining the checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, y, epochs=2, batch_size=128, callbacks=callbacks_list) # the number of epochs causes the machine to learn more and make less errors... but also takes time
filename = "weights-improvement-50-1.2850--bigger.hdf5" # loading network weights using the checkpoint
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars)) # creating a reverse mapping to print letters instead of numbers

start = np.random.randint(0, len(dataX)-1) # gotta start somewhere! (getting the random seed)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

for i in range(500):                      # generating characters from the seed
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone Duh Done Done DOOOOOOOONE!")                          # printing the results
