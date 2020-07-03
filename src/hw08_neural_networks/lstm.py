from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D
import sys
import numpy as np

from hw08_neural_networks import get_data

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10

def build_and_evaluate_model(x_train, y_train, x_dev, y_dev):
    '''Builds, trains and evaluates a keras LSTM model.'''
    return 0, 0, None # TODO: REMOVE for exercise 3.
    x_train = None # TODO: Exercise 3.1
    x_dev = None # TODO: Exercise 3.1
    y_train = None # TODO: Exercise 3.1
    y_dev = None # TODO: Exercise 3.1
    model = Sequential()
    # TODO: Ex. 3.2 - 3.4
    # Add layers.
    # Compile model.
    # Fit to data.
    # ODOT
    score, acc = model.evaluate(x_dev, y_dev)
    return score, acc, model

def main(argv):
    print('Loading data...')
    x_train, y_train, x_dev, y_dev, word2id = get_data.nltk_data(vocab_size=VOCAB_SIZE)
    print(len(x_train), 'training samples')
    print(len(x_dev), 'development samples')
    score, acc, _ = build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
    print('\ndev score:', score)
    print('dev accuracy:', acc)

if __name__ == "__main__":
    main(sys.argv[1:])
