import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pickle


def scale_variables_independently(df, exclude_columns, scaler_type):
    """
    Scales the variables in a DataFrame independently using the specified scaler.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        exclude_columns (list): List of columns to exclude from scaling.
        scaler_type (str): The type of scaler to use ('MinMaxScaler', 'StandardScaler', 'RobustScaler', 'QuantileTransformer').
        
    Returns:
        scaled_df (pd.DataFrame): The scaled DataFrame.
        scalers_dict (dict): A dictionary with the scaler for each scaled variable.
    """
    # Check for valid scaler type
    scaler_classes = {
        "MinMaxScaler": MinMaxScaler,
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
        "QuantileTransformer": QuantileTransformer
    }
    
    if scaler_type not in scaler_classes:
        raise ValueError(f"Invalid scaler_type. Choose from {list(scaler_classes.keys())}")
    
    # Initialize dictionary to store scalers
    scalers_dict = {}
    
    # Create a copy of the DataFrame to avoid modifying the original
    scaled_df = df.copy()
    
    # Identify columns to scale
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    
    # Apply scaling to each column independently
    for column in columns_to_scale:
        scaler = scaler_classes[scaler_type]()
        scaled_values = scaler.fit_transform(df[[column]])
        
        # Update the scaled DataFrame
        scaled_df[column] = scaled_values.flatten()
        
        # Store the scaler in the dictionary
        scalers_dict[column] = scaler
    
    return scaled_df, scalers_dict

df = pd.read_pickle('RAIN_NOWCAST_FRAMEWORK/Data/Processed/final_df_4h.pkl')
exclude_columns = ['lat', 'lon', 'validdate', 'is_rain_1h:idx']

scaler_type = "RobustScaler"
scaled_df, scalers_dict = scale_variables_independently(df, exclude_columns, scaler_type)
print(scaled_df)

scaled_df.to_pickle(f"RAIN_NOWCAST_FRAMEWORK/Data/Processed/df_scaled_{scaler_type}_4h.pkl")

with open(f"RAIN_NOWCAST_FRAMEWORK/Data/Scalers/scalers/scalers_{scaler_type}_4h.pkl", "wb") as f:
    pickle.dump(scalers_dict, f)

print("SCALERS AND DATAFRAME SAVED")