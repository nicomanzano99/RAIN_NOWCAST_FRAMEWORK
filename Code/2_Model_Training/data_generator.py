import numpy as np
import os
from tensorflow.keras.utils import Sequence
import tensorflow as tf


class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, shuffle=True, include_index=False, filter_params=None, target_type='regression'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_index = include_index
        self.filter_params = filter_params
        self.target_type = target_type 


        # Load file list
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")])
        self.indexes = np.arange(len(self.files))

        # Initialize variables for loading
        self.current_file_index = 0
        self.X = None
        self.y = None
        self.index_data = None
        self.X_params = None
        self.load_next_file()

        # Shuffle the data initially
        if self.shuffle:
            self.on_epoch_end()

    def load_next_file(self):
        """Loads the next file into memory and filters variables if filter_params is provided."""
        
        if self.current_file_index >= len(self.files):
            self.current_file_index = 0  

        npz_file = self.files[self.current_file_index]

        try:
            data = np.load(npz_file, allow_pickle=True)
            self.X = data["X"]
            self.y = data["y"]
            self.index_data = data.get("TIME_LAT_LON_INDEX")
            self.X_params = data.get("X_params")  # Get parameter names
        except Exception as e:
            print(f"Error loading file {npz_file}: {e}")
        
        self.current_file_index += 1

        if self.filter_params:
            self.filter_variables()

    def filter_variables(self):
        """Filters the input variables based on the filter_params dictionary."""
        if not self.X_params.size:
            raise ValueError("X_params is not available in the dataset for filtering.")

        mask = [self.filter_params.get(param, False) for param in self.X_params]
        mask = np.array(mask, dtype=bool)

        self.X = self.X[:, :, mask]
        self.X_params = [param for param, include in zip(self.X_params, mask) if include]

    def __len__(self):
        """Number of batches per epoch."""
        total_samples = sum([np.load(f, allow_pickle=True)["X"].shape[0] for f in self.files])
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, batch_index):
        """Generates one batch of data."""
        
        if self.X is None or batch_index * self.batch_size >= self.X.shape[0]:
            self.load_next_file()

        start = (batch_index * self.batch_size) % self.X.shape[0]
        end = start + self.batch_size

        if end > self.X.shape[0]:
            end = self.X.shape[0]


        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        index_batch = self.index_data[start:end] if self.include_index else None

        # Ensure y_batch is reshaped correctly depending on target type
        if self.target_type == "regression":
            y_batch = y_batch[:, 0]
            y_batch = y_batch.reshape(-1, 1) 
        elif self.target_type == "classification":
            num_classes = len(np.unique(y_batch))
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes)  

        
        if end == self.X.shape[0]:  
            self.load_next_file()

        # Check if the batch sizes match
        if X_batch.shape[0] != y_batch.shape[0]:
            raise ValueError(f"Batch size mismatch: X_batch has {X_batch.shape[0]} samples, "
                             f"but y_batch has {y_batch.shape[0]} samples.")

        if self.include_index:
            return X_batch, y_batch, index_batch
        else:
            return X_batch, y_batch

    def on_epoch_end(self):
        """Shuffles the data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.files = [self.files[i] for i in self.indexes]
