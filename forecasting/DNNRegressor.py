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
