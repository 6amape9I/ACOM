import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical


def MLP():

    model = Sequential()
    model.add(Dense(128, input_shape=(28 * 28,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = MLP()

    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy}')

    predictions = model.predict(x_test)

    for i in range(10):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}')
        plt.show()
