"""
DNN Classifier.
"""
import numpy as np
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

import data


class NN(tf.keras.Model):
    def __init__(self):
        super(NN, self).__init__()
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        # Output layer.
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.d1(x)
        X = self.drop1(x)
        x = self.d2(x)
        return self.out(x)


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
    test_loss.update_state(loss)
    test_accuracy.update_state(y, pred)


if __name__ == "__main__":
    # Hyper-parameters
    EPOCHS = 100
    BATCH_SIZE = 2048
    LR = 1e-5
    print("Tenserflow version: ", tf.__version__)
    # Prepare Data
    df = data.load_whole("./data/")
    X, y = data.gen_sup(df)
    X = X.astype(np.float32)
    X, y = map(lambda z: z.values, (X, y))
    y = y.reshape(-1, 1)

    # Create Polynomial Features
    poly = preprocessing.PolynomialFeatures(degree=3)
    X = poly.fit_transform(X)


    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=None, shuffle=True)

    # Standardize features
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("X_train@", X_train.shape)
    print("X_test@", X_test.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(int(1e6)).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(BATCH_SIZE)

    model = NN()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(
        name="test_accuracy")

    trace = {"train": [], "val": []}
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        # Loop over batches.
        for x, y in train_ds:
            # x @ (batch_size, num_features)
            # y @ (batch_size, 1) --> probit
            train_step(x, y)

        for t_x, t_y in test_ds:
            test_step(t_x, t_y)

        if epoch % 5 == 0:
            report = "Epoch {:d}, Loss: {:0.6f}, Accuracy: {:0.6f}, Validation Loss: {:0.6f}, Validation Accuracy: {:0.6f}"
            print(report.format(
                epoch+1,
                train_loss.result(),
                train_accuracy.result()*100,
                test_loss.result(),
                test_accuracy.result()*100))

        # Record loss
        trace["train"].append(train_loss.result())
        trace["val"].append(test_loss.result())

    # AUC
    pred_train = model(X_train).numpy()
    pred_test = model(X_test).numpy()

    auc_train = metrics.roc_auc_score(y_true=y_train, y_score=pred_train)
    auc_test = metrics.roc_auc_score(y_true=y_test, y_score=pred_test)

    print(f"AUC on Training Set: {auc_train: 0.6f}")
    print(f"AUC on Testing Set: {auc_test: 0.6f}")

    plt.plot(np.log(trace["train"]))
    plt.plot(np.log(trace["val"]))
    plt.xlabel("Epochs")
    plt.ylabel("Log Cross Entropy Loss")
    plt.legend(["Training", "Validation"])
    plt.title(f"LR={LR}, AUC_train={auc_train:0.3f}, AUC_test={auc_test:0.3f}")
    plt.show()
