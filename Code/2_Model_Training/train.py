import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from custom_losses import CustomLosses

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check and configure GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found")


import wandb

# Ensure any active W&B run is finished
wandb.finish()

import os
os.environ["WANDB_DIR"] = "RAIN_NOWCAST_FRAMEWORK/Models"
os.environ["WANDB_MODE"] = "online"

from architectures import Architectures
from data_generator import DataGenerator 

# Base directories
BASE_MODEL_DIR = "RAIN_NOWCAST_FRAMEWORK/Models"
TRAINING_LOG_PATH = os.path.join(BASE_MODEL_DIR, "training_log.csv")
BALANCING_LOG_PATH = "RAIN_NOWCAST_FRAMEWORK/ML_data/balancing_log.csv"

# Sweep configuration
sweep_config = {
    "method": "random",  
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "num_layers": {"value": 2},  # Número de capas LSTM
        "learning_rate": {"value": 0.0039232},
        "batch_size": {"values": [64]},
        "hidden_units_1": {"values": [512]},  # Unidades para la primera capa
        "hidden_units_2": {"values": [256]},  # Segunda capa (si aplica)
        "dropout": {"value": 0.25268},
        "epochs": {"value": 200},
    },
}

# Sweep configuration
sweep_config = {
    "method": "bayes",  
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "num_layers": {"values": [1, 2, 3]},  # Número de capas LSTM
        "learning_rate": {"min": 0.0001, "max": 0.005},
        "batch_size": {"values": [64, 128]},
        "hidden_units_1": {"values": [128, 256, 512]},  # Unidades para la primera capa
        "hidden_units_2": {"values": [64, 128, 256]},  # Segunda capa (si aplica)
        "hidden_units_3": {"values": [32, 64, 128]},  # Tercera capa (si aplica)
        "dropout": {"min": 0.1, "max": 0.3},
        "epochs": {"value": 200},
    },
}


# Function to dynamically create directories for training
def create_training_directories(balancing_technique, dataset_id):
    dataset_folder = os.path.join(BASE_MODEL_DIR, f"{balancing_technique}_{dataset_id}")
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(dataset_folder, run_time)

    logs_dir = os.path.join(run_folder, "logs")
    models_dir = os.path.join(run_folder, "models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    return run_folder, logs_dir, models_dir


# Log sweep experiments to CSV
def log_to_csv_sweep(config, experiment_dir, logs_dir, final_model_path, training_time, time_per_epoch, history, architecture_name, variables_dict, target_type):
    final_training_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1

    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sweep_Run_ID": wandb.run.id,
        "Architecture": architecture_name,
        "Learning_Rate": config.learning_rate,
        "Batch_Size": config.batch_size,
        "Hidden_Units": config.hidden_units,
        "Dropout": config.dropout,
        "Epochs": config.epochs,
        "Training_Time": training_time,
        "Time_per_Epoch": time_per_epoch,
        "Final_Training_Loss": final_training_loss,
        "Final_Validation_Loss": final_val_loss,
        "Best_Validation_Loss": best_val_loss,
        "Best_Epoch": best_epoch,
        "Experiment_Dir": experiment_dir,
        "Logs_Dir": logs_dir,
        "Final_Model_Path": final_model_path,
        "Variables_Dict": variables_dict,
        "Target_Type": target_type  # Added target type (regression/classification)
    }

    if not os.path.exists(TRAINING_LOG_PATH):
        pd.DataFrame([log_data]).to_csv(TRAINING_LOG_PATH, index=False)
    else:
        training_log = pd.read_csv(TRAINING_LOG_PATH)
        training_log = training_log.append(log_data, ignore_index=True)
        training_log.to_csv(TRAINING_LOG_PATH, index=False)

    print(f"Trial logged to {TRAINING_LOG_PATH}")


def get_input_shape(npz_dir, filter_params=None):
    if filter_params is None:
        filter_params = {}  # Default to empty dictionary if no filter_params are passed
    
    first_file = os.path.join(npz_dir, os.listdir(npz_dir)[0])  # Get the first .npz file
    data = np.load(first_file, allow_pickle=True)

    # The shape after filtering
    input_shape = data['X'].shape[1:]  # Exclude the batch size dimension
    return input_shape


def get_loss_and_activation(target_type, num_classes, loss_name="mse"):
    if target_type == "regression":
        # Get custom loss dynamically
        loss_function = getattr(CustomLosses, loss_name, CustomLosses.mse)()
        return loss_function, "linear", ["mae"]
    elif target_type == "classification":
        if num_classes == 2:
            return "binary_crossentropy", "sigmoid", ["accuracy"]  # Binary classification
        else:
            return "categorical_crossentropy", "softmax", ["accuracy"]  # Multi-class classification
    else:
        raise ValueError("Unknown target_type. Use 'regression' or 'classification'.")


# Get number of categories from CSV Categories field
def get_num_categories_from_csv(dataset_id, balancing_technique):
    balancing_log = pd.read_csv(BALANCING_LOG_PATH)
    row = balancing_log[(balancing_log["ID"] == dataset_id) & (balancing_log["Balancing Method"] == balancing_technique)]
    if row.empty:
        raise ValueError(f"No matching dataset found for ID={dataset_id} and Balancing Method={balancing_technique}")
    
    categories = eval(row.iloc[0]["Categories"])
    return len(categories)  # Number of categories


# Training function
def train(balancing_technique, dataset_id, architecture_name, variables_dict, target_type):
    print(f"Training started with ID={dataset_id}, Balancing Technique={balancing_technique}, Architecture={architecture_name}")
    print("Initializing W&B...")

    wandb.init(project="precipitation-prediction", dir="RAIN_NOWCAST_FRAMEWORK/Models")
    print("W&B Initialized successfully.")
    print("W&B initialized, getting config...")

    config = wandb.config
    print(f"Config loaded: {config}")
    experiment_dir, logs_dir, models_dir = create_training_directories(balancing_technique, dataset_id)

    # Get input shape from the CSV file
    balancing_log = pd.read_csv(BALANCING_LOG_PATH)
    row = balancing_log[(balancing_log["ID"] == dataset_id) & (balancing_log["Balancing Method"] == balancing_technique)]
    if row.empty:
        raise ValueError(f"No matching dataset found for ID={dataset_id} and Balancing Method={balancing_technique}")

    # Fetch dataset path from the CSV
    dataset_path = row.iloc[0]["Balanced Dataset Path"]
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "validation")
    
    # Get input shape from the first .npz file in the train directory
    input_shape = get_input_shape(train_path, filter_params=None)
    
    if TARGET_TYPE == "classification":
        # Get number of categories from the CSV
        num_classes = get_num_categories_from_csv(dataset_id, balancing_technique)
        
    elif TARGET_TYPE == "regression":
        num_classes = 1
        
    else:
        raise ValueError("Unknown target_type. Use 'regression' or 'classification'.")
        

    # Get loss, activation, and metrics based on target type and number of categories
    loss, activation, metrics = get_loss_and_activation(target_type, num_classes, loss_name="mse")
    model_constructor = getattr(Architectures, architecture_name)

    
    if architecture_name == "transformer":
        model = model_constructor(
            input_shape=input_shape, 
            num_heads=config.num_heads,  
            ff_units=config.ff_units,  
            num_layers=config.num_layers,  
            output_units=num_classes,  
        )
    elif architecture_name == "cnn_1d":
        model = model_constructor(
        input_shape=input_shape,  
        conv_filters=config.conv_filters,  
        kernel_size=config.kernel_size,  
        pool_size=config.pool_size, 
        dense_units=config.dense_units, 
        dropout=config.dropout,
        output_units=num_classes, 
        output_activation=activation
        )
    
    elif architecture_name == "lstm_fine_tuning":
        model = model_constructor(
            input_shape = input_shape, 
            num_layers=config.num_layers, 
            hidden_units_1=config.hidden_units_1, 
            hidden_units_2=config.hidden_units_2, 
            dropout=config.dropout
        )

    else:
        model = model_constructor(
            input_shape=input_shape,
            hidden_units=config.hidden_units,
            dropout=config.dropout,
            output_units=num_classes,
            output_activation=activation
        )
        

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss=loss, metrics=metrics)
    

    balancing_log = pd.read_csv(BALANCING_LOG_PATH)

    row = balancing_log[(balancing_log["ID"] == dataset_id) & (balancing_log["Balancing Method"] == balancing_technique)]
    if row.empty:
        raise ValueError(f"No matching dataset found for ID={dataset_id} and Balancing Method={balancing_technique}")

    train_path = os.path.join(row.iloc[0]["Balanced Dataset Path"], "train")
    val_path = os.path.join(row.iloc[0]["Balanced Dataset Path"], "validation")
    pca_applied = row.iloc[0]["PCA Applied"]

    train_gen = DataGenerator(train_path, batch_size=config.batch_size, filter_params=None if pca_applied else variables_dict, target_type=target_type)
    val_gen = DataGenerator(val_path, batch_size=config.batch_size, filter_params=None if pca_applied else variables_dict, target_type=target_type)


    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(models_dir, "checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    best_model_callback = ModelCheckpoint(
        filepath=os.path.join(models_dir, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1)

    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    wandb_callback = wandb.keras.WandbCallback(save_model=False)

    start_training = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=[
            checkpoint_callback,
            best_model_callback,
            tensorboard_callback,
            early_stopping_callback,
            reduce_lr_callback,
            wandb_callback
        ]
    )
    end_training = time.time()
    training_time = end_training - start_training
    time_per_epoch = training_time / config.epochs

    final_model_path = os.path.join(models_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    log_to_csv_sweep(
        config=config,
        experiment_dir=experiment_dir,
        logs_dir=logs_dir,
        final_model_path=final_model_path,
        training_time=training_time,
        time_per_epoch=time_per_epoch,
        history=history,
        architecture_name=architecture_name,
        variables_dict=variables_dict,
        target_type=target_type  # Log the selected target type
    )

    wandb.finish()

# Main script
if __name__ == "__main__":
    BALANCING_TECHNIQUE = "weights"
    DATASET_ID = 15
    ARCHITECTURE_NAME = "lstm_fine_tuning"

    VARIABLES_DICT = {
        "precip_1h:mm": True,
        "prob_precip_1h:p": True,
        "cloud_liquid_water:kgm2": True,
        "super_cooled_liquid_water:kgm2": True,
        "is_rain_1h:idx": False, 
        "relative_humidity_2m:p": True,
        "relative_humidity_850hPa:p": True,
        "mixing_ratio_850hPa:kgkg": True,
        "dew_point_2m:K": True,
        "sfc_pressure:hPa": True,
        "geopotential_height_850hPa:m": True,
        "geopotential_height_700hPa:m": True,
        "geopotential_height_500hPa:m": False,
        "layer_thickness_700hPa_1000hPa:m": False,
        "wind_speed_10m:ms": True,
        "wind_dir_10m:d": True,
        "total_cloud_cover:p": False,
        "cape:Jkg": False,
        "lifted_index:K": False,
        "clear_sky_rad:W": False,
        "direct_rad:W": True,
        "global_rad:W": False
    
    }
    
    TARGET_TYPE = "regression"  # or "classification"

    sweep_id = wandb.sweep(sweep_config, project="Precipitation_6h_final_LSTM")
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=lambda: train(BALANCING_TECHNIQUE, DATASET_ID, ARCHITECTURE_NAME, VARIABLES_DICT, TARGET_TYPE), count=1)
