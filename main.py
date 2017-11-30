from parser import load
from nn import NN

import numpy as np
epoch_count = 100

if __name__ == '__main__':
    print('Creating network model...')
    nn = NN()

    nn.train(load('./vindi/'), epoch=1)
