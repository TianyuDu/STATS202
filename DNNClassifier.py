"""
DNN Classifier.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

import data_proc


class NN(tf.keras.Model):
    def __init__(self, num_neurons):
        super(NN, self).__init__()
        self.drop1 = tf.keras.layers.Dropout(0.8)
        self.dense1 = tf.keras.layers.Dense(num_neurons[0], activation="relu")
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_neurons[1], activation="relu")
        self.drop3 = tf.keras.layers.Dropout(0.5)
        # Output layer.
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.drop2(x)
        x = self.dense2(x)
        x = self.drop3(x)
        return self.out(x)


def main(
        get_data: callable,
        EPOCHS: int = 10,
        PERIOD: int = 5,
        BATCH_SIZE: int = 256,
        LR: float = 1e-5,
        NEURONS: list = [128, 128],
        forecast: bool = False,
        tuning: bool = False,
) -> None:
    """
    Main Training Process for DNN classifier
    # TODO: Write doc string.
    """
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = loss_object(y, pred)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        train_loss.update_state(loss)
        train_accuracy.update_state(y, pred)


    @tf.function
    def test_step(x, y):
        # Test and validation step have the same operation.
        pred = model(x)
        loss = loss_object(y, pred)
        dev_loss.update_state(loss)
        dev_accuracy.update_state(y, pred)

    print("Reading data...")
    X_train, X_dev, y_train, y_dev, X_test = get_data()
    print(f"X_train@{X_train.shape}, X_dev@{X_dev.shape}")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(int(1e6)).batch(BATCH_SIZE)

    dev_ds = tf.data.Dataset.from_tensor_slices(
        (X_dev, y_dev)).batch(BATCH_SIZE)

    num_fea = X_train.shape[1]
    model = NN(num_neurons=NEURONS)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="train_accuracy")

    dev_loss = tf.keras.metrics.Mean(name="dev_loss")
    dev_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="dev_accuracy")

    trace = {"train": [], "val": []}
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        dev_loss.reset_states()
        dev_accuracy.reset_states()
        # Loop over batches.
        for x, y in train_ds:
            # x @ (batch_size, num_features)
            # y @ (batch_size, 1) --> probit
            train_step(x, y)

        for t_x, t_y in dev_ds:
            test_step(t_x, t_y)

        if (epoch+1) % PERIOD == 0:
            report = "Epoch {:d}, Loss: {:0.6f}, Accuracy: {:0.6f}, Validation Loss: {:0.6f}, Validation Accuracy: {:0.6f}"
            print(report.format(
                epoch+1,
                train_loss.result(),
                train_accuracy.result()*100,
                dev_loss.result(),
                dev_accuracy.result()*100))

        # Record loss
        trace["train"].append(train_loss.result())
        trace["val"].append(dev_loss.result())

    # AUC
    pred_train = model(X_train).numpy()
    pred_dev = model(X_dev).numpy()

    auc_train = metrics.roc_auc_score(y_true=y_train, y_score=pred_train)
    auc_dev = metrics.roc_auc_score(y_true=y_dev, y_score=pred_dev)

    print(f"AUC on Training Set: {auc_train: 0.6f}")
    print(f"AUC on Developing Set: {auc_dev: 0.6f}")

    if forecast:
        pred = model(X_test)
        return pred.numpy()
    if tuning:
        return {
            "AUC_TRAIN": auc_train,
            "AUC_DEV": auc_dev,
            "LOSS_TRAIN": train_loss.result(),
            "LOSS_DEV": dev_loss.result(),
            "ACCURACY_TRAIN": train_accuracy.result(),
            "ACCURACY_DEV": dev_accuracy.result(),
        }

    plt.plot(np.log(trace["train"]))
    plt.plot(np.log(trace["val"]))
    plt.xlabel("Epochs")
    plt.ylabel("Log Cross Entropy Loss")
    plt.legend(["Training", "Validation"])
    plt.title(f"LR={LR}, AUC_train={auc_train:0.3f}, AUC_dev={auc_dev:0.3f}")
    plt.show()
