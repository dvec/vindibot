from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout


class NN:
    _DROPOUT = 0.02
    
    _model = Sequential([
        Dense(820, input_shape=[820], activation='tanh'),
        Dropout(_DROPOUT),
        Dense(820, activation='tanh'),
        Dropout(_DROPOUT),
        Dense(100, activation='tanh'),
        Dropout(_DROPOUT),
        Activation('sigmoid'),
        Dense(5),
    ])

    def __init__(self):
        self._model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def train(self, x, y, epoch=30, verbose=1):
        self._model.fit(x, y, epochs=epoch, batch_size=1, verbose=verbose)

    def run(self, x, verbose=1):
        return self._model.predict_classes(x, verbose=verbose)

    def save(self, filepath='model.h5'):
        self._model.save(filepath)

    def load(self, filepath='model.h5'):
        self._model.load_weights(filepath)
