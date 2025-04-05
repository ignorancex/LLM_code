import tensorflow.keras as keras


# categorical crossentropy model
def get_categorical_model(num_classes, input_shape):
    # returns multi-class model with given number of classes

    assert len(input_shape) == 2 and input_shape[1] == 1, "Invalid input shape."

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv1D(8, kernel_size=5, padding="same", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def get_binary_model(input_shape):
    # returns binary model

    assert len(input_shape) == 2 and input_shape[1] == 1, "Invalid input shape."

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv1D(8, kernel_size=5, padding="same", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
