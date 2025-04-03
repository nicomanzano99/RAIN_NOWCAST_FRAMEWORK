import os
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import time


class PrecipitationDataProcessor:
    def __init__(self, data_dir, scaler_dir, temporal_window, thresholds, balance_strategy,
                 train_test_split, validation_strategy, sequence_base_dir):
        self.data_dir = data_dir
        self.scaler_dir = scaler_dir
        self.temporal_window = temporal_window
        self.thresholds = thresholds
        self.balance_strategy = balance_strategy
        self.train_test_split = train_test_split
        self.validation_strategy = validation_strategy
        self.sequence_base_dir = sequence_base_dir
        self.total_samples = {"train": 0, "validation": 0, "test": 0}

        self.scaled_df = None
        self.scalers = None
        self.pca_model = None

    def select_dataset(self, scaler_type, resolution):
        df_pattern = f"df_scaled_{scaler_type}_{resolution}.pkl"
        df_path = os.path.join(self.data_dir, df_pattern)
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Scaled DataFrame not found for scaler {scaler_type} and resolution {resolution}.")
        self.scaled_df = pd.read_pickle(df_path)
        print(f"Loaded scaled DataFrame: {df_path}")

        scaler_pattern = f"scalers_{scaler_type}_{resolution}.pkl"
        scaler_path = os.path.join(self.scaler_dir, scaler_pattern)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scalers not found for scaler {scaler_type} and resolution {resolution}.")
        with open(scaler_path, "rb") as f:
            self.scalers = pickle.load(f)
        print(f"Loaded scalers: {scaler_path}")

    def reduce_thresholds(self, thresholds, n_categories):
        """
        Reduces the number of categories by combining adjacent thresholds.

        Parameters:
            thresholds (dict): Dictionary of thresholds with scaled values.
            n_categories (int): Desired number of categories.

        Returns:
            dict: Reduced thresholds with the specified number of categories.
        """
        # Sort thresholds by their scaled values
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])

        # Extract values and names
        category_names = [t[0] for t in sorted_thresholds]
        category_values = [t[1] for t in sorted_thresholds]

        # Split the scaled values into `n_categories` evenly
        new_bins = np.linspace(min(category_values), max(category_values), n_categories + 1)

        # Assign labels to new bins
        new_thresholds = {}
        for i in range(n_categories):
            lower_bound = new_bins[i]
            upper_bound = new_bins[i + 1]
            new_thresholds[f"Category {i + 1}"] = (lower_bound, upper_bound)

        return new_thresholds

    def calculate_thresholds(self, resolution, n_categories=None):
        """
        Dynamically calculate thresholds for classification based on temporal resolution
        and optionally reduce the number of categories.

        Parameters:
            resolution (str): Temporal resolution (e.g., '1h', '4h', etc.).
            n_categories (int): Desired number of categories (optional).

        Returns:
            dict: Thresholds adjusted for temporal resolution and optionally reduced categories.
        """
        hourly_factor = {"1h": 24, "4h": 6, "6h": 4, "8h": 3}[resolution]
        raw_scaled_thresholds = {k: v / hourly_factor for k, v in self.thresholds.items()}

        if 'precip_obs' not in self.scalers:
            raise KeyError("'precip_obs' scaler not found in the loaded scalers. Ensure it's included.")

        precip_scaler = self.scalers['precip_obs']
        scaled_thresholds = {
            k: precip_scaler.transform([[v]])[0, 0] for k, v in raw_scaled_thresholds.items()
        }

        # Reduce thresholds if n_categories is specified
        if n_categories:
            scaled_thresholds = self.reduce_thresholds(scaled_thresholds, n_categories)

        print(f"Scaled thresholds for {resolution}: {scaled_thresholds}")
        
        return scaled_thresholds
    
    def find_dataset_folder(self, scaler_type, resolution):
        """
        Finds or creates the correct folder for the current dataset based on the scaler and resolution.

        Parameters:
            scaler_type (str): The type of scaler (e.g., 'MinMaxScaler').
            resolution (str): The temporal resolution (e.g., '1h').

        Returns:
            str: Path to the dataset folder.
        """
        base_folder = os.path.join(self.sequence_base_dir, f"{scaler_type}_{resolution}")
        dataset_id = 1
        while os.path.exists(f"{base_folder}_{dataset_id:02d}"):
            dataset_id += 1
        dataset_folder = f"{base_folder}_{dataset_id:02d}"
        os.makedirs(dataset_folder, exist_ok=True)
        return dataset_folder

    def apply_pca(self, sequences, fit=True):
        """
        Applies PCA to reduce the dimensionality of the input sequences, maintaining the timesteps.

        Parameters:
            sequences (np.ndarray): The input data with shape (n_samples, timesteps, features).
            fit (bool): If True, fits the PCA model on the data. If False, transforms using the existing PCA model.

        Returns:
            np.ndarray: The PCA-transformed sequences with shape (n_samples, timesteps, reduced_features).
        """
        n_samples, timesteps, features = sequences.shape  # Extract the original shape

        # Initialize storage for PCA-transformed data
        transformed_data = []

        for t in range(timesteps):  # Loop through each timestep
            timestep_data = sequences[:, t, :]  # Extract data for this timestep (n_samples, features)

            if fit:
                # Fit PCA and calculate the number of components for 95% variance
                pca = PCA()
                pca.fit(timestep_data)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = next(i for i, var in enumerate(cumulative_variance) if var >= 0.95) + 1

                # Store the PCA model for this timestep
                if not hasattr(self, "pca_models"):
                    self.pca_models = {}  # Create storage for PCA models
                self.pca_models[t] = PCA(n_components=n_components)

                # Apply PCA transformation
                timestep_transformed = self.pca_models[t].fit_transform(timestep_data)
            else:
                # Use the existing PCA model for this timestep
                timestep_transformed = self.pca_models[t].transform(timestep_data)

            transformed_data.append(timestep_transformed)

        # Concatenate PCA-transformed data along the timestep axis
        reduced_features = transformed_data[0].shape[1]  # Reduced features dimension
        transformed_data = np.stack(transformed_data, axis=1)  # Shape: (n_samples, timesteps, reduced_features)

        return transformed_data


    def assign_classification_label(self, value):
        """
        Assigns a classification label to a regression value based on the thresholds.

        Parameters:
            value (float): The regression value (precipitation observation).

        Returns:
            int: The classification category as an integer.
        """
        for category, (lower_bound, upper_bound) in self.thresholds.items():
            if lower_bound <= value < upper_bound:
                return int(category.split(" ")[-1]) if category.startswith("Category") else category
        return -1  # Default if no category matches
    
    

    def create_sequences(self, df):
        """
        Creates sequences of data for the temporal window, dynamically calculating regression
        and classification targets.

        Parameters:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            np.ndarray: The sequences (n_samples, timesteps, features).
            np.ndarray: The targets (regression and classification).
            np.ndarray: The time, lat, and lon index.
        """
        sequences = []
        targets = []
        time_lat_lon_index = []

        for idx in range(len(df) - len(self.temporal_window) + 1):
            seq_indices = [idx + t for t in self.temporal_window]
            if all(i >= 0 and i < len(df) for i in seq_indices):
                # Extract sequence for the current window
                seq = df.iloc[seq_indices].drop(columns=["validdate", "lat", "lon", "precip_obs"]).values
                sequences.append(seq)

                # Calculate regression target (current timestep `t=0`)
                reg_target = df.iloc[idx + self.temporal_window.index(0)]["precip_obs"]

                # Calculate classification target using thresholds
                class_target = self.assign_classification_label(reg_target)

                # Append targets
                targets.append([reg_target, class_target])

                # Append time, lat, and lon for the current sequence
                time_lat_lon_index.append([
                    df.iloc[idx + self.temporal_window.index(0)]["validdate"],
                    df.iloc[idx + self.temporal_window.index(0)]["lat"],
                    df.iloc[idx + self.temporal_window.index(0)]["lon"]
                ])

        # Convert lists to numpy arrays
        sequences = np.array(sequences)  # Shape: (n_samples, timesteps, features)
        targets = np.array(targets)      # Shape: (n_samples, 2)
        time_lat_lon_index = np.array(time_lat_lon_index)  # Shape: (n_samples, 3)

        return sequences, targets, time_lat_lon_index



    def save_sequences(self, dataset_folder, split, station, X, y, X_params, Y_params, time_lat_lon_index):
        """
        Saves sequences into train/validation/test subdirectories.

        Parameters:
            dataset_folder (str): Path to the dataset folder.
            split (str): One of 'train', 'validation', or 'test'.
            station (str): Station identifier.
            X (np.ndarray): Features.
            y (np.ndarray): Targets.
            X_params (list): List of NWP parameters.
            Y_params (list): List of target parameters.
            time_lat_lon_index (np.ndarray): Array of timestamp, latitude, and longitude.

        Returns:
            None
        """
        split_folder = os.path.join(dataset_folder, split)
        os.makedirs(split_folder, exist_ok=True)
        np.savez_compressed(
            os.path.join(split_folder, f"{station}.npz"),
            X=X,
            y=y,
            X_params=X_params,
            Y_params=Y_params,
            TIME_LAT_LON_INDEX=time_lat_lon_index
        )
        print(f"Saved {split} sequences for station: {station}")
        
    def load_csv(self, csv_path):
        """
        Loads an existing CSV file if it exists, otherwise returns an empty DataFrame.

        Parameters:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame or an empty DataFrame.
        """
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            # Return an empty DataFrame if the file doesn't exist
            return pd.DataFrame()

    def log_to_csv(self, scaler_type, resolution, dataset_folder, sequence_window, categories, validation_strategy, pca):
        """
        Logs details about the processed dataset to an Excel file.

        Parameters:
            scaler_type (str): The type of scaler used.
            resolution (str): Temporal resolution (e.g., '1h', '4h', etc.).
            dataset_folder (str): Path to the dataset folder.
            sequence_window (list): Temporal sequence window.
            categories (dict): Categories used for classification.
            validation_strategy (str): Validation strategy ('percentage' or 'last_week').
            pca (bool): Whether PCA was applied.

        Returns:
            None
        """
        csv_path = os.path.join(self.sequence_base_dir, "processing_log.csv")
        data = {
            "ID": [len(self.load_csv(csv_path)) + 1 if os.path.exists(csv_path) else 1],  # Auto-increment ID
            "Scaler": [scaler_type],
            "Resolution": [resolution],
            "Sequence Window": [str(sequence_window)],
            "PCA Applied": [pca],  # New column for PCA
            "Total Samples": [self.total_samples],  # Total number of samples (to be set during processing)
            "Path": [dataset_folder],
            "Categories": [str(categories)],
            "Validation Strategy": [validation_strategy],
            "Stations": [len(self.scaled_df['lat'].unique())],  # Number of unique stations
        }

        # Create or update the Excel file
        df = pd.DataFrame(data)
        
        # Append to the CSV file if it exists, otherwise create a new file
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        

        print(f"Processing details logged to {csv_path}")


    
    def process(self, scaler_type, resolution, n_categories=None, pca=False, test_date=None):
        """
        Orchestrates the entire processing workflow, creating station IDs from lat and lon,
        dynamically adjusting thresholds, and splitting data into train, validation, and test.

        Parameters:
            scaler_type (str): The type of scaler to use (e.g., 'MinMaxScaler', 'RobustScaler').
            resolution (str): Temporal resolution (e.g., '1h', '4h', etc.).
            n_categories (int): Desired number of categories (optional).
            pca (bool): Whether to apply PCA transformation.
            test_date (str): The date used to split data into training/validation and testing (e.g., '2022-01-01').

        Returns:
            None
        """
        # Select dataset and scaler
        self.select_dataset(scaler_type, resolution)

        # cutoff_year = 2021  # Define the cutoff year
        # original_size = len(self.scaled_df)
        # self.scaled_df = self.scaled_df[self.scaled_df["validdate"].dt.year <= cutoff_year]
        # print(f"Filtered dataset from {original_size} samples to {len(self.scaled_df)} samples (cutoff year: {cutoff_year})")


        # Calculate thresholds dynamically and ensure the last category extends to infinity
        self.thresholds = self.calculate_thresholds(resolution, n_categories=n_categories)
        last_category = list(self.thresholds.keys())[-1]  # Get the last category
        lower_bound, _ = self.thresholds[last_category]   # Extract the lower bound
        self.thresholds[last_category] = (lower_bound, float('inf'))  # Set upper bound to infinity

        # Dynamically determine NWP parameters (X_params)
        excluded_columns = ['validdate', 'lat', 'lon', 'precip_obs']
        X_params = [col for col in self.scaled_df.columns if col not in excluded_columns]

        # Create station dictionary: {station_id: (lat, lon)}
        unique_stations = self.scaled_df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
        station_dict = {i: (row['lat'], row['lon']) for i, row in unique_stations.iterrows()}

        # Convert test_date to timezone-aware datetime if provided
        if test_date:
            test_date = pd.to_datetime(test_date).tz_localize("UTC")  # Ensure test_date is UTC timezone-aware

        # Find the dataset folder
        dataset_folder = self.find_dataset_folder(scaler_type, resolution)

        print(f"Processing {len(station_dict)} stations...")
        # Process data for each station
        for station_id, (lat, lon) in tqdm(station_dict.items(), desc="Processing Stations"):
            
            station_start_time = time.time()  # Start timer for this station
            # Filter data for the current station
            station_df = self.scaled_df[
                (self.scaled_df['lat'] == lat) & (self.scaled_df['lon'] == lon)
            ]

            # Drop rows with NaN values in the target column ('precip_obs')
            station_df = station_df.dropna(subset=['precip_obs'])

            # Split into train/validation and test based on test_date
            if test_date:
                train_val_df = station_df[station_df['validdate'] < test_date]
                test_df = station_df[station_df['validdate'] >= test_date]
            else:
                train_val_df = station_df
                test_df = pd.DataFrame()

            # Split train/validation using the specified validation strategy
            if self.validation_strategy == "percentage":
                # Use the last 'train_test_split' percentage of train_val_df for validation
                val_size = int(len(train_val_df) * self.train_test_split)
                train_df = train_val_df.iloc[:-val_size]
                val_df = train_val_df.iloc[-val_size:]

            elif self.validation_strategy == "last_week":
                # Dynamically split train/validation by taking the last week of each month
                train_df = []
                val_df = []
                for month, month_df in train_val_df.groupby(train_val_df['validdate'].dt.to_period('M')):
                    last_week = month_df['validdate'].max() - pd.Timedelta(days=7)
                    val_mask = month_df['validdate'] >= last_week
                    train_mask = month_df['validdate'] < last_week

                    train_df.append(month_df[train_mask])
                    val_df.append(month_df[val_mask])

                # Concatenate train and validation dataframes
                train_df = pd.concat(train_df)
                val_df = pd.concat(val_df)

            # Create sequences for training and validation
            train_sequences, train_targets, train_index = self.create_sequences(train_df)
            val_sequences, val_targets, val_index = self.create_sequences(val_df)

            # Create sequences for testing if test_df is not empty
            test_sequences, test_targets, test_index = ([], [], [])
            if not test_df.empty:
                test_sequences, test_targets, test_index = self.create_sequences(test_df)
                
            # Increment sample counts
            self.total_samples["train"] += len(train_sequences)
            self.total_samples["validation"] += len(val_sequences)
            self.total_samples["test"] += len(test_sequences)

            # Apply PCA only to train and transform validation/test using the same model
            if pca:
                train_sequences = self.apply_pca(train_sequences, fit=True)  # Fit PCA on train
                val_sequences = self.apply_pca(val_sequences, fit=False)    # Transform val
                if len(test_sequences) > 0:
                    test_sequences = self.apply_pca(test_sequences, fit=False)  # Transform test

            # # Balance the dataset for train sequences
            # train_sequences, train_targets = self.balance_data(train_sequences, train_targets)

            # Save train, validation, and test data
            self.save_sequences(
                dataset_folder, "train", f"station_{station_id}", train_sequences, train_targets, X_params,
                ["regression", "classification"], train_index
            )
            self.save_sequences(
                dataset_folder, "validation", f"station_{station_id}", val_sequences, val_targets, X_params,
                ["regression", "classification"], val_index
            )
            if len(test_sequences) > 0:
                self.save_sequences(
                    dataset_folder, "test", f"station_{station_id}", test_sequences, test_targets, X_params,
                    ["regression", "classification"], test_index
                )
                
            station_end_time = time.time()
            print(f"Station {station_id} processed in {station_end_time - station_start_time:.2f} seconds.")

                
        #After processing and saving sequences:
        self.log_to_csv(
            scaler_type=scaler_type,
            resolution=resolution,
            dataset_folder=dataset_folder,
            sequence_window=self.temporal_window,
            categories=self.thresholds,
            validation_strategy=self.validation_strategy,
            pca=pca,  # Include PCA information
        )



# Define the input parameters
data_dir = "RAIN_NOWCAST_FRAMEWORK/Data/Processed"
scaler_dir = "RAIN_NOWCAST_FRAMEWORK/Data/Scalers"
sequence_base_dir = "RAIN_NOWCAST_FRAMEWORK/ML_data"

temporal_window = [-2, -1, 0] 

#temporal_window = [-4, -2, 0, 2, 4]

thresholds = {
    "No Rain": 0.0,
    "Very Light Rain": 0.9,
    "Light Rain": 10.0,
    "Moderate Rain": 30.0,
    "Heavy Rain": 70.0,
    "Very Heavy Rain": 150.0,
    "Extremely Heavy Rain": 151.0,
}
balance_strategy = "oversample"  # Oversampling for balance
train_test_split = 0.2  # 20% of training/validation for validation
validation_strategy = "last_week"  # Use the last week of each month for validation

# Instantiate the PrecipitationDataProcessor class
processor = PrecipitationDataProcessor(
    data_dir=data_dir,
    scaler_dir=scaler_dir,
    temporal_window=temporal_window,
    thresholds=thresholds,
    balance_strategy=balance_strategy,
    train_test_split=train_test_split,
    validation_strategy=validation_strategy,
    sequence_base_dir=sequence_base_dir
)


# Call the process method
processor.process(
    scaler_type="RobustScaler",  # Choose the scaler (e.g., MinMaxScaler, RobustScaler)
    resolution="6h",            # Temporal resolution (e.g., '1h', '4h', '6h', '8h')
    n_categories=4,             # Number of desired categories for classification (optional)
    pca=True,                   # Apply PCA transformation (True to enable)
    test_date="2024-01-01"      # Define the date to split train/val and test
)




