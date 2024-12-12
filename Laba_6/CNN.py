import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical


def CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def deep_CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    models = [CNN(), deep_CNN()]
    exstra_models = [MLP()]

    for i, model in enumerate(models):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f'Training Model {i + 1}')
        history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_split=0.2)

        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print(f'Model {i + 1} Test accuracy: {test_accuracy}')

    for i, model in enumerate(exstra_models):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f'Training Model {i + 1}')
        history = model.fit(x_train.reshape(60000, 784), y_train, epochs=10, batch_size=32,
                            validation_split=0.2)

        test_loss, test_accuracy = model.evaluate(x_test.reshape(10000, 784), y_test)
        print(f'Model {i + 1} Test accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
