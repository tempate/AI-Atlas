from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


SEED = 0
HIDDEN_LAYERS = [256, 128]
ACTIVATION = 'relu'
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5
VAL_SIZE = 0.1

NUM_CLASSES = 10
SAMPLE_SHAPE = (28, 28, 1)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32').reshape((-1, *SAMPLE_SHAPE)) / 255.0
    x_test = x_test.astype('float32').reshape((-1, *SAMPLE_SHAPE)) / 255.0
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    inputs = Input(shape=SAMPLE_SHAPE)
    x = Flatten()(inputs)
    for units in HIDDEN_LAYERS:
        x = Dense(units, activation=ACTIVATION)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='mnist')
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy'],
    )
    model.summary()
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
    )
    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)


if __name__ == "__main__":
    main()
