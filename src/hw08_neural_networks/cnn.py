from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Convolution1D, GlobalMaxPooling1D
import sys
import numpy as np

from hw08_neural_networks import get_data

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10

def build_and_evaluate_model(x_train, y_train, x_dev, y_dev):
    '''Builds, trains and evaluates a keras LSTM model.'''
    # return 0, 0, None # TODO: REMOVE for exercise 4.
    x_train = sequence.pad_sequences(x_train, MAX_LEN)
    x_dev = sequence.pad_sequences(x_dev, MAX_LEN)
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    # ODOT
    model = Sequential()
    # TODO: Ex. 4.2 - 4.4
    model.add(Embedding(VOCAB_SIZE, 50))
    model.add(Convolution1D(filters=25, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, BATCH_SIZE, EPOCHS)
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
