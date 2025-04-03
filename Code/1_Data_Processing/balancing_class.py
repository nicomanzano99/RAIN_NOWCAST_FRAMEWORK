import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import json


class DatasetBalancer:
    def __init__(self, processing_log_path):
        """
        Initializes the DatasetBalancer.

        Parameters:
            processing_log_path (str): Path to the processing log CSV file.
        """
        self.processing_log_path = processing_log_path

    def read_processing_log(self, dataset_id):
        """
        Reads the processing log to fetch metadata for the specified dataset ID.

        Parameters:
            dataset_id (int): The ID of the dataset to balance.

        Returns:
            dict: Metadata for the dataset.
        """
        df = pd.read_csv(self.processing_log_path)
        dataset_row = df[df["ID"] == dataset_id]
        if dataset_row.empty:
            raise ValueError(f"Dataset with ID {dataset_id} not found in the processing log.")
        return dataset_row.iloc[0].to_dict()

    def balance_dataset(self, dataset_id, balancing_method, balancing_params=None):
        """
        Balances the dataset based on the specified method.

        Parameters:
            dataset_id (int): The ID of the dataset to balance.
            balancing_method (str): The balancing method ('oversampling' or 'proportional').
            balancing_params (dict, optional): Parameters for proportional balancing (e.g., category percentages).

        Returns:
            None
        """
        # Read metadata from the processing log
        metadata = self.read_processing_log(dataset_id)
        dataset_path = metadata["Path"]  # Base directory of the original dataset
        categories = eval(metadata["Categories"], {"inf": math.inf})   # Convert string representation back to a dictionary
        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "validation")

        # Create output folders dynamically based on the dataset path and balancing method
        balancing_folder = os.path.join(dataset_path, balancing_method)  # Add balancing method to base path
        os.makedirs(balancing_folder, exist_ok=True)
        train_output_path = os.path.join(balancing_folder, "train")
        val_output_path = os.path.join(balancing_folder, "validation")
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(val_output_path, exist_ok=True)

        # Handle the case for "weights" balancing method
        if balancing_method == "weights":
            print("Computing class weights and shuffling data without balancing...")
            
            # Load train data
            train_sequences, train_targets, train_index = self.load_npz_files(train_path)

            # Compute class distribution
            class_counts = {}
            for target in train_targets[:, 1]:  # Assuming the second column contains classification targets
                class_counts[target] = class_counts.get(target, 0) + 1

            # Compute class weights
            total_samples = sum(class_counts.values())
            class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

            # Shuffle train and validation data
            train_sequences, train_targets, train_index = self.shuffle_data(train_sequences, train_targets, train_index)
            val_sequences, val_targets, val_index = self.load_npz_files(val_path)
            val_sequences, val_targets, val_index = self.shuffle_data(val_sequences, val_targets, val_index)

            # Save the shuffled train and validation data
            self.save_balanced_data(train_sequences, train_targets, train_index, train_output_path)
            self.save_balanced_data(val_sequences, val_targets, val_index, val_output_path)

            # Save class weights to a JSON file
            weights_path = os.path.join(balancing_folder, "class_weights.json")
            with open(weights_path, "w") as f:
                json.dump(class_weights, f, indent=4)

            # Log balancing operation
            self.log_balancing(
                dataset_id=dataset_id,
                balancing_method=balancing_method,
                balancing_params=balancing_params,
                output_path=balancing_folder,
            )

            print(f"Class weights computed and saved to {weights_path}.")
            #return class_weights
        else:
            # For other balancing methods: oversampling or proportional
            print(f"Balancing train data with {balancing_method}...")
            train_sequences, train_targets, train_index = self.load_npz_files(train_path)
            balanced_sequences, balanced_targets, balanced_index = self.apply_balancing(
                train_sequences, train_targets, train_index, balancing_method, balancing_params, categories
            )
            
            # Shuffle balanced train data
            balanced_sequences, balanced_targets, balanced_index = self.shuffle_data(
                balanced_sequences, balanced_targets, balanced_index
            )
            
            self.save_balanced_data(balanced_sequences, balanced_targets, balanced_index, train_output_path)

            # Process validation data
            print("Processing validation data (shuffling only)...")
            val_sequences, val_targets, val_index = self.load_npz_files(val_path)
            shuffled_sequences, shuffled_targets, shuffled_index = self.shuffle_data(val_sequences, val_targets, val_index)
            self.save_balanced_data(shuffled_sequences, shuffled_targets, shuffled_index, val_output_path)

            print(f"Balanced dataset saved at {balancing_folder}.")

            # Log balancing operation
            self.log_balancing(
                dataset_id=dataset_id,
                balancing_method=balancing_method,
                balancing_params=balancing_params,
                output_path=balancing_folder,
            )


    def load_npz_files(self, folder_path):
        """
        Loads sequences, targets, and time_lat_lon_index from all NPZ files in the folder.

        Parameters:
            folder_path (str): Path to the folder containing NPZ files.

        Returns:
            tuple: Loaded sequences, targets, and time_lat_lon_index.
        """
        sequences, targets, time_lat_lon_index = [], [], []
        for npz_file in tqdm(sorted(os.listdir(folder_path)), desc="Loading NPZ files"):
            if npz_file.endswith(".npz"):
                data = np.load(os.path.join(folder_path, npz_file), allow_pickle=True)
                sequences.append(data["X"])
                targets.append(data["y"])
                time_lat_lon_index.append(data["TIME_LAT_LON_INDEX"])
                self.X_params = data['X_params']
        return np.vstack(sequences), np.vstack(targets), np.vstack(time_lat_lon_index)

    def apply_balancing(self, sequences, targets, time_lat_lon_index, method, params, categories):
        """
        Balances the dataset based on the specified method.

        Parameters:
            sequences (np.ndarray): Input sequences.
            targets (np.ndarray): Input targets.
            time_lat_lon_index (np.ndarray): Time, latitude, and longitude for each sequence.
            method (str): Balancing method ('oversampling' or 'proportional').
            params (dict): Parameters for the balancing method.
            categories (dict): Target categories.

        Returns:
            tuple: Balanced sequences, targets, and time_lat_lon_index.
        """
        if method == "oversampling":
            balanced_sequences, balanced_targets, balanced_time_lat_lon = [], [], []
            for category in np.unique(targets[:, 1]):  # Classification target
                category_mask = targets[:, 1] == category
                category_sequences = sequences[category_mask]
                category_targets = targets[category_mask]
                category_index = time_lat_lon_index[category_mask]

                max_samples = max([sum(targets[:, 1] == c) for c in np.unique(targets[:, 1])])
                repetitions = max_samples // len(category_sequences)
                remainder = max_samples % len(category_sequences)

                balanced_sequences.extend(np.tile(category_sequences, (repetitions, 1, 1)))
                balanced_sequences.extend(category_sequences[:remainder])
                balanced_targets.extend(np.tile(category_targets, (repetitions, 1)))
                balanced_targets.extend(category_targets[:remainder])
                balanced_time_lat_lon.extend(np.tile(category_index, (repetitions, 1)))
                balanced_time_lat_lon.extend(category_index[:remainder])
            return (
                np.array(balanced_sequences),
                np.array(balanced_targets),
                np.array(balanced_time_lat_lon),
            )
        
        elif method == "undersampling":
            balanced_sequences, balanced_targets, balanced_time_lat_lon = [], [], []
            min_samples = min([sum(targets[:, 1] == c) for c in np.unique(targets[:, 1])])

            for category in np.unique(targets[:, 1]):
                category_mask = targets[:, 1] == category
                category_sequences = sequences[category_mask]
                category_targets = targets[category_mask]
                category_index = time_lat_lon_index[category_mask]

                sampled_indices = np.random.choice(len(category_sequences), min_samples, replace=False)

                balanced_sequences.extend(category_sequences[sampled_indices])
                balanced_targets.extend(category_targets[sampled_indices])
                balanced_time_lat_lon.extend(category_index[sampled_indices])

            return (
                np.array(balanced_sequences),
                np.array(balanced_targets),
                np.array(balanced_time_lat_lon),
            )
        
        elif method == "proportional":
            if not params:
                raise ValueError("Proportional balancing requires percentage parameters.")
            total_samples = len(sequences)
            balanced_sequences, balanced_targets, balanced_time_lat_lon = [], [], []
            for category, percentage in params.items():
                category_mask = targets[:, 1] == int(category)
                category_sequences = sequences[category_mask]
                category_targets = targets[category_mask]
                category_index = time_lat_lon_index[category_mask]
                num_samples = int(total_samples * percentage)
                balanced_sequences.extend(category_sequences[:num_samples])
                balanced_targets.extend(category_targets[:num_samples])
                balanced_time_lat_lon.extend(category_index[:num_samples])
            return (
                np.array(balanced_sequences),
                np.array(balanced_targets),
                np.array(balanced_time_lat_lon),
            )
        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def shuffle_data(self, sequences, targets, time_lat_lon_index):
        """
        Shuffles the data.

        Parameters:
            sequences (np.ndarray): Input sequences.
            targets (np.ndarray): Input targets.
            time_lat_lon_index (np.ndarray): Time, latitude, and longitude for each sequence.

        Returns:
            tuple: Shuffled sequences, targets, and time_lat_lon_index.
        """
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        return sequences[indices], targets[indices], time_lat_lon_index[indices]

    def log_balancing(self, dataset_id, balancing_method, balancing_params, output_path):
        """
        Logs details about the balancing operation to the balancing log.

        Parameters:
            dataset_id (int): The ID of the dataset being balanced.
            balancing_method (str): The balancing method used ('oversampling' or 'proportional').
            balancing_params (dict): Parameters for proportional balancing (if applicable).
            output_path (str): Path to the balanced dataset.

        Returns:
            None
        """
        # Define the path for the balancing log
        balancing_log_path = "RAIN_NOWCAST_FRAMEWORK/ML_data/balancing_log.csv"

        # Load the processing log to get the original dataset row
        df_processing_log = pd.read_csv(self.processing_log_path)
        dataset_row = df_processing_log[df_processing_log["ID"] == dataset_id]
        if dataset_row.empty:
            raise ValueError(f"Dataset with ID {dataset_id} not found in the processing log.")

        # Add new balancing columns
        dataset_row["Balancing Method"] = balancing_method
        dataset_row["Balancing Parameters"] = str(balancing_params) if balancing_params else "N/A"
        dataset_row["Balanced Dataset Path"] = output_path

        # Append to the balancing log
        if os.path.exists(balancing_log_path):
            # Append to existing file
            dataset_row.to_csv(balancing_log_path, mode="a", header=False, index=False)
        else:
            # Create a new file
            dataset_row.to_csv(balancing_log_path, mode="w", header=True, index=False)

        print(f"Balancing details logged to {balancing_log_path}")

        
    
    def save_balanced_data(self, sequences, targets, time_lat_lon_index, output_path, chunk_size_mb=100):
        """
        Saves balanced sequences, targets, and time_lat_lon_index to NPZ files in chunks.

        Parameters:
            sequences (np.ndarray): Input sequences.
            targets (np.ndarray): Input targets.
            time_lat_lon_index (np.ndarray): Time, latitude, and longitude for each sequence.
            output_path (str): Path to save the balanced data.
            chunk_size_mb (int): Maximum size of each NPZ file in MB.

        Returns:
            None
        """
        # Estimate max samples per file
        sequence_size = sequences.shape[1] * sequences.shape[2] * 4  # 4 bytes per float
        target_size = targets.shape[1] * 4  # 4 bytes per float
        time_lat_lon_size = time_lat_lon_index.shape[1] * 8  # 8 bytes for datetime and floats
        sample_size = sequence_size + target_size + time_lat_lon_size

        max_samples_per_file = (chunk_size_mb * 1024 * 1024) // sample_size
        num_files = int(np.ceil(len(sequences) / max_samples_per_file))

        for i in range(num_files):
            start_idx = i * max_samples_per_file
            end_idx = min(start_idx + max_samples_per_file, len(sequences))

            file_path = os.path.join(output_path, f"chunk_{i + 1}.npz")
            np.savez_compressed(
                file_path,
                X=sequences[start_idx:end_idx],
                y=targets[start_idx:end_idx],
                TIME_LAT_LON_INDEX=time_lat_lon_index[start_idx:end_idx],
                X_params=self.X_params
            )
            

balancer = DatasetBalancer(
    processing_log_path="RAIN_NOWCAST_FRAMEWORK/ML_data/processing_log.csv"
)

balancer.balance_dataset(
    dataset_id=1, 
    balancing_method="weights"
)