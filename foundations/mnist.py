from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from library.tensor.MLP import MLP
from library.tensor.Tensor import Tensor
import numpy as np


LEARNING_RATE = 0.05
BATCH_SIZE = 64
N_EPOCHS = 50
VAL_SIZE = 0.1


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the input
    x_train = x_train.astype('float32').reshape((-1, 28*28)) / 255.0
    x_test = x_test.astype('float32').reshape((-1, 28*28)) / 255.0

    # Categorize the target
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


def fit(model, x_train, y_train):
    loss = ((Tensor(y_train) - model(x_train))**2).mean()

    # Compute the gradient of each parameter in the model
    model.zero_grad()
    loss.backward()

    # Update the parameters with gradient descent
    for param in model.parameters():
        param.data -= LEARNING_RATE * param.grad

    return loss


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = MLP(x_train.shape[1], [32, 32, 10])

    # Train the model
    for it in range(N_EPOCHS):
        n_batches = x_train.shape[0] // BATCH_SIZE

        losses = []
        for i in range(n_batches):
            x_train_ = x_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
            y_train_ = y_train[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
            loss = fit(model, x_train_, y_train_)
            losses.append(loss.data)

        print(f"{it+1}. Loss: {np.mean(losses):.3f}")

    # Evaluate the model
    accuracy = 0
    for x, y in zip(x_test, y_test):
        if np.argmax(model(x).data) == np.argmax(y):
            accuracy += 1
    accuracy /= len(x_test)

    print("Test accuracy:", accuracy)


if __name__ == "__main__":
    main()
