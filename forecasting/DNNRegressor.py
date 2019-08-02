"""
DNNRegressor for the forecasting task.
Aug. 1, 2019
"""
import tensorflow as tf


def build_regressor(
        num_inputs: int,
        num_neurons: List[int],
        input_dropout: float = 0.2,
        internal_dropout: float = 0.5,
) -> tf.keras.Model:
    """
    Builds a DNN regressor for forecasting tasks.
    """
    inputs = tf.keras.Input(shape=(num_inputs, ), name="input_layer")
    x = tf.keras.layers.Dropout(input_dropout)(inputs)
    # **** Hidden Layers ****
    for i, neuron in enumerate(num_neurons):
        x = tf.keras.layers.Dense(
            units=neuron, activation="relu", name="dense_layer_"+str(i))(x)
        x = tf.keras.layers.Dropout(internal_dropout)(x)
    # **** Output Layer ****
    outputs = tf.keras.layers.Dense(
        units=1, activation="linear", name="output_layer")(x)
    return tf.keras.Model(inputs, outputs)


def main(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        EPOCHS: int = 10,
        PERIOD: int = 5,
        BATCH_SIZE: int = 32,
        LR: float = 1e-5,
        NEURONS: list = [128, 128],
        forecast: bool = False,
        tuning: bool = True,
) -> Union[dict, np.ndarray]:
    model = build_regressor(
        num_inputs=X_train.shape[1],
        num_neurons=NEURONS
    )
    if tuning:
        val_ratio = 0.2
        verbose = 1
    else:
        val_ratio = 0.0
        verbose = 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="mean_squared_error",
    )
    hist = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=verbose,
        validation_split=val_ratio,
    )
    if forecast:
        return model.predict(X_test)

    if tuning:
        step = 50  # report step
        record_lst = list()
        for t in range(1, EPOCHS // step):
            # Report training histories.
            record = {"EPOCHS": t * step, "BATCH_SIZE": BATCH_SIZE,
                      "LR": LR, "NEURONS": NEURONS}
            losses = {d: v[t * step] for d, v in hist.history.items()}
            record.update(losses)
            record_lst.append(record)
        # For the final result.
        final_record = {
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "NEURONS": NEURONS,
        }
        # Retrive the final loss
        losses = {d: v[-1] for d, v in hist.history.items()}
        final_record.update(losses)
        record_lst.append(final_record)
        return record_lst
