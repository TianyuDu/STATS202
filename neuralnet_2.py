"""
The naive NN approach, use PANSS scores only.
"""
import tensorflow as tf
from sklearn import model_selection
import numpy as np

import data


class NN(tf.keras.Model):
    def __init__(self):
        super(NN, self).__init__()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(32, activation="relu")
        # Output layer.
        self.out = tf.keras.layers.Dense(1, activation="linear")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(y, pred)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))


if __name__ == "__main__":
    print(tf.__version__)
    df = data.load_whole("./data/")
    X, y = data.gen_sup(df)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=None, shuffle=True)

    model = NN()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    model = NN()
    model.compile(optimizer, loss=loss_object, metrics=["accuracy"])
    model.fit(
        x=X_train.values,
        y=y_train.values,
        batch_size=32,
        epochs=30,
        verbose=1,
        validation_split=0.2
    )
    test_loss = model.evaluate(x=X_test, y=y_test)
    print(f"Test set result: cross-entropy loss {test_loss[0]: 0.6f}, accuracy {test_loss[1]: 0.6f}")
