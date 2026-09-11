"""
lstm_autoencoder.py -- the LSTM Autoencoder architecture.

Input(seq_len, n_features) -> LSTM(64) -> LSTM(32, latent) -> RepeatVector
                            -> LSTM(32) -> LSTM(64) -> TimeDistributed(Dense)

Kept deliberately small: 2 encoder + 2 decoder LSTM layers, no attention, no
bidirectional layers. Step 5/7 experiments (see experiment_log.csv) showed no
evidence that added complexity would help on this dataset/fault mix -- in
fact a simple rolling z-score statistical baseline matched or beat this
architecture on overall F1 (see reports/evaluation_report.md "Baseline
Comparison"), so growing the network further was not justified by evidence.
"""
import tensorflow as tf


def build_model(seq_len, n_features, learning_rate=1e-3):
    inp = tf.keras.Input(shape=(seq_len, n_features), name="input_sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="encoder_lstm_64")(inp)
    x = tf.keras.layers.LSTM(32, return_sequences=False, name="encoder_lstm_32_latent")(x)
    x = tf.keras.layers.RepeatVector(seq_len, name="repeat_vector")(x)
    x = tf.keras.layers.LSTM(32, return_sequences=True, name="decoder_lstm_32")(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="decoder_lstm_64")(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features), name="reconstruction")(x)
    model = tf.keras.Model(inp, out, name=f"lstm_autoencoder_seq{seq_len}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model
