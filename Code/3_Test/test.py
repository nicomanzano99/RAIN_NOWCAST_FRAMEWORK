import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_test_data_path(dataset_id, balancing_technique, balancing_log_path):
    """
    Obtiene la ruta del dataset de test eliminando la subcarpeta 'oversampling' del path.
    """
    balancing_log = pd.read_csv(balancing_log_path)
    row = balancing_log[(balancing_log["ID"] == dataset_id) & (balancing_log["Balancing Method"] == balancing_technique)]
    
    if row.empty:
        raise ValueError(f"No se encontró el dataset con ID={dataset_id} y técnica de balanceo {balancing_technique}")
    
    dataset_path = row.iloc[0]["Balanced Dataset Path"] 

    base_path = os.path.dirname(dataset_path) 

    # Construir la ruta de test
    test_path = os.path.join(base_path, "test")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró la carpeta de test en {test_path}")

    return test_path


# Función para obtener el modelo guardado
def get_best_model_path(experiment_path):
    """
    Busca el modelo con nombre 'best_model.h5' dentro de la carpeta 'models'.
    """
    models_dir = os.path.join(experiment_path, "models")
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de modelos en {models_dir}")

    best_model_path = os.path.join(models_dir, "best_model.h5")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No se encontró 'best_model.h5' en {models_dir}")

    return best_model_path


# Función para realizar predicciones
def make_predictions(dataset_id, balancing_technique, experiment_path, scalers_path, balancing_log_path, variables_dict, pca=False, batch_size=128):
    """
    Function to make predictions given the experiment path
    """
    print(f"Loading data for ID={dataset_id}, tecnique={balancing_technique}...")
    test_path = get_test_data_path(dataset_id, balancing_technique, balancing_log_path)
    
    print(f"Loading scalers from {scalers_path}...")
    
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    target_var = "precip_obs"  
    if target_var not in scalers:
        raise KeyError(f"No scaler for {target_var} at {scalers_path}")

    target_scaler = scalers[target_var]

    print(f"Loading model from {experiment_path}...")
    best_model_path = get_best_model_path(experiment_path)
    model = tf.keras.models.load_model(best_model_path, compile=False)

    test_files = sorted(os.listdir(test_path))
    test_files = [os.path.join(test_path, f) for f in test_files if f.endswith(".npz")]

    if not test_files:
        raise FileNotFoundError("No test files in the given directory")

    all_predictions = []
    all_observations = []
    all_timestamps = []
    all_lat_values = []
    all_lon_values = []

    if not pca:

        selected_variables = {k for k, v in variables_dict.items() if v} 

        first_npz = np.load(test_files[0], allow_pickle=True)
        X_params = first_npz["X_params"] 

        valid_indices = [i for i, var in enumerate(X_params) if var in selected_variables]


    print(f"Selected variables for test: {selected_variables}")
    print(f"Available variables for test: {X_params}")

    if len(valid_indices) != len(selected_variables):
        print("The amount of avriables is not the same")


    # Cargar y predecir
    for npz_file in test_files:
        data = np.load(npz_file, allow_pickle=True)
        if not pca:
            X_test = data["X"][:, :, valid_indices]
        else:
           X_test = data["X"]
           
        Y_test = data["y"][:, 0]
        timestamps = data["TIME_LAT_LON_INDEX"][:,0]
        lat_values = data["TIME_LAT_LON_INDEX"][:,1]
        lon_values = data["TIME_LAT_LON_INDEX"][:,2]

        predictions = model.predict(X_test, batch_size=batch_size).flatten()

        predictions_descaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        observations_descaled = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        all_predictions.extend(predictions_descaled)
        all_observations.extend(observations_descaled)
        all_timestamps.extend(timestamps)
        all_lat_values.extend(lat_values)
        all_lon_values.extend(lon_values)

    results_df = pd.DataFrame({
        "Timestamp": all_timestamps,
        "Latitude": all_lat_values,
        "Longitude": all_lon_values,
        "Prediction": all_predictions,
        "Observation": all_observations
    })

    print("Predictions completed")
    return results_df


if __name__ == "__main__":

    DATASET_ID = 7
    BALANCING_TECHNIQUE = "weights"
    EXPERIMENT_PATH = "RAIN_NOWCAST_FRAMEWORK/Models/model_folder"
    SCALERS_PATH = "RAIN_NOWCAST_FRAMEWORK/Data/Scalers/scalers_pickle.pkl" 
    BALANCING_LOG_PATH = "RAIN_NOWCAST_FRAMEWORK/ML_data/balancing_log.csv"
    PCA = False

    VARIABLES_DICT = {
        "precip_1h:mm": True,
        "prob_precip_1h:p": True,
        "cloud_liquid_water:kgm2": True,
        "super_cooled_liquid_water:kgm2": True,
        "is_rain_1h:idx": True, 
        "relative_humidity_2m:p": True,
        "relative_humidity_850hPa:p": True,
        "mixing_ratio_850hPa:kgkg": True,
        "dew_point_2m:K": True,
        "sfc_pressure:hPa": True,
        "geopotential_height_850hPa:m": True,
        "geopotential_height_700hPa:m": True,
        "geopotential_height_500hPa:m": True,
        "layer_thickness_700hPa_1000hPa:m": True,
        "wind_speed_10m:ms": True,
        "wind_dir_10m:d": True,
        "total_cloud_cover:p": True,
        "cape:Jkg": True,
        "lifted_index:K": True,
        "clear_sky_rad:W": True,
        "direct_rad:W": True,
        "global_rad:W": True
    
    }


    results_df = make_predictions(DATASET_ID, BALANCING_TECHNIQUE, EXPERIMENT_PATH, SCALERS_PATH, BALANCING_LOG_PATH, VARIABLES_DICT, PCA)

    output_csv = os.path.join(EXPERIMENT_PATH, "filename.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved at {output_csv}")
