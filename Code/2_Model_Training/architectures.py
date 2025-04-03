from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout, TimeDistributed, GRU, Bidirectional, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import tensorflow as tf


class Architectures:
    """
    A class to define and manage various model architectures.
    """

    @staticmethod
    def lstm(input_shape, hidden_units=128, dropout=0.2, output_units=1, output_activation="linear"):
        """
        Defines an LSTM architecture.

        Parameters:
            input_shape (tuple): Shape of the input (timesteps, features).
            hidden_units (int): Number of units in the LSTM layer.
            dropout (float): Dropout rate for regularization.
            output_units (int): Number of output units.
            output_activation (str): Activation function for the output layer.

        Returns:
            model (Sequential): Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(output_units, activation=output_activation))

        model.summary()
        return model
    

    @staticmethod
    def lstm_fine_tuning(input_shape, num_layers=1, hidden_units_1=256, hidden_units_2=128,
        dropout=0.2, output_units=1, output_activation="linear"):
        """
        Defines an LSTM architecture.

        Parameters:
            input_shape (tuple): Shape of the input (timesteps, features).
            hidden_units (int): Number of units in the LSTM layer.
            dropout (float): Dropout rate for regularization.
            output_units (int): Number of output units.
            output_activation (str): Activation function for the output layer.

        Returns:
            model (Sequential): Compiled LSTM model.
        """
        model = Sequential()

        model.add(LSTM(hidden_units_1, input_shape=input_shape, return_sequences=(num_layers > 1)))

        if num_layers > 1:
            model.add(LSTM(hidden_units_2, return_sequences=(num_layers > 2)))


        model.add(Dropout(dropout))  # Regularización
        model.add(Dense(output_units, activation=output_activation))  # Capa de salida

        model.summary()
        return model

    @staticmethod
    def gru(input_shape, hidden_units=128, dropout=0.2, output_units=1, output_activation="linear"):
        """
        Defines a GRU architecture.

        Parameters:
            input_shape (tuple): Shape of the input (timesteps, features).
            hidden_units (int): Number of units in the GRU layer.
            dropout (float): Dropout rate for regularization.
            output_units (int): Number of output units.
            output_activation (str): Activation function for the output layer.

        Returns:
            model (Sequential): Compiled GRU model.
        """
        model = Sequential()
        model.add(GRU(hidden_units, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(output_units, activation=output_activation))

        model.summary()
        return model

    @staticmethod
    def bidirectional_lstm(input_shape, hidden_units=128, dropout=0.2, output_units=1, output_activation="linear"):
        """
        Defines a Bidirectional LSTM architecture.

        Parameters:
            input_shape (tuple): Shape of the input (timesteps, features).
            hidden_units (int): Number of units in the LSTM layer.
            dropout (float): Dropout rate for regularization.
            output_units (int): Number of output units.
            output_activation (str): Activation function for the output layer.

        Returns:
            model (Sequential): Compiled Bidirectional LSTM model.
        """
        model = Sequential()
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(output_units, activation=output_activation))

        model.summary()
        return model


    @staticmethod
    def cnn_1d(input_shape, conv_filters=32, kernel_size=3, pool_size=2, dense_units=128, dropout=0.2, output_units=1, output_activation="linear"):
        """
        Defines a 1D CNN architecture.

        Parameters:
            input_shape (tuple): Shape of the input (timesteps, features).
            conv_filters (int): Number of filters in the convolutional layer.
            kernel_size (int): Kernel size for the convolutional layer.
            pool_size (int): Pool size for max pooling.
            dense_units (int): Number of units in the dense layer.
            dropout (float): Dropout rate for regularization.
            output_units (int): Number of output units.
            output_activation (str): Activation function for the output layer.

        Returns:
            model (Sequential): Compiled CNN1D model.
        """
        model = Sequential()
        # Add Conv1D layer with padding to ensure no dimensions go negative
        model.add(layers.Conv1D(conv_filters, kernel_size, activation="relu", input_shape=input_shape, padding="same"))
        model.add(layers.MaxPooling1D(pool_size=pool_size))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(output_units, activation=output_activation))
        
        return model

    @staticmethod
    def transformer(input_shape, num_heads=8, ff_units=128, num_layers=2, output_units=1):
        inputs = layers.Input(shape=input_shape)

        # Initial embedding
        x = layers.Dense(ff_units)(inputs)  

        for _ in range(num_layers):
            # Self-Attention Block
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x) 

            ff_output = layers.Dense(ff_units, activation="relu")(x)
            
            ff_output = layers.Dense(int(x.shape[-1]))(ff_output) 

            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)

        outputs = layers.Dense(output_units)(x)

        model = tf.keras.Model(inputs, outputs)
        return model



    
