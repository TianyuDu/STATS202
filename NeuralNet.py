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
        self.d1 = tf.keras.layers.Dense(256, activation="sigmoid")
        # self.drop1 = tf.keras.layers.Dropout(0.2)
        self.d2 = tf.keras.layers.Dense(512, activation="sigmoid")
        # Output layer.
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.d1(x)
        # X = self.drop1(x)
        x = self.d2(x)
        return self.out(x)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(y, pred)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, pred)


@tf.function
def test_step(x, y):
    # Test and validation step have the same operation.
    pred = model(x)
    loss = loss_object(y, pred)
    test_loss(loss)
    test_accuracy(y, pred)


if __name__ == "__main__":
    print(tf.__version__)
    df = data.load_whole("./data/")
    X, y = data.gen_sup(df)
    X = X.astype(np.float32)
    X, y = map(lambda z: z.values, (X, y))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=None, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(32)

    model = NN()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="test_accuracy")

    EPOCHS = 100
    for epoch in range(EPOCHS):
        for x, y in train_ds:
            train_step(x, y)

        for t_x, t_y in test_ds:
            test_step(t_x, t_y)

        report = "Epoch {:d}, Loss: {:0.6f}, Accuracy: {:0.6f}, Validation Loss: {:0.6f}, Validation Accuracy: {:0.6f}"
        print(report.format(
            epoch+1,
            train_loss.result(),
            train_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100)
        )
