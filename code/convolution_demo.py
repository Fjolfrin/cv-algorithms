from convolution import *
from matplotlib import pyplot as plt
import numpy as np
from warnings import filterwarnings

# Do not show any warnings that do not affect the result.
filterwarnings('ignore')

def conv1d_demo(x, h):
    # Calculate the output of the convolution between the two signals,
    # and print the result.
    y = conv1d(x, h, padding="same", stride=1)
    print(y)

    # Plot the two input signals and their convolution.
    fig, ax = plt.subplots(3, sharex=True)

    ax[0].plot(range(len(x)), x, 'b')
    ax[0].title.set_text("Input signal")
    ax[1].plot(range(len(h)), h, 'g')
    ax[1].title.set_text("Applied filter")
    ax[2].plot(range(y.shape[0]), y, 'r')
    ax[2].title.set_text("Convolution result")

    plt.show()

def conv2d_demo(x, h):
    # Calculate the output of the convolution between the two signals,
    # and print the result.
    y = conv2d(x, h, padding="full", stride=1)
    print(y)

    # Plot the two input signals and their convolution.
    fig, ax = plt.subplots(3, sharex=True)

    ax[0].imshow(x)
    ax[0].title.set_text("Input signal")
    ax[1].imshow(h)
    ax[1].title.set_text("Applied filter")
    ax[2].imshow(y)
    ax[2].title.set_text("Convolution result")

    plt.show()

x = np.arange(25).reshape((5, 5))
h = np.arange(9).reshape((3, 3))
conv2d_demo(x, h)