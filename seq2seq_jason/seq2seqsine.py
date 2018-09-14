from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# generate a sequence of random integers
def generate_sequence(length, f=2, Fs=100):
    fs = 100  # sample rate
    f = 2  # the frequency of the signal

    x = np.arange(length)
    y = np.sin(2 * np.pi * f * x / Fs)
    return x, y


x1, y1 = generate_sequence(1000, f=2, Fs=100)
plt.plot(x1,y1)
plt.show()