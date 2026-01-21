import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.utils import generate_index
from darts.dataprocessing.transformers import Scaler
import streamlit as st
from notionhelper import NotionHelper
from darts.models import NBEATSModel
from darts.metrics import smape

def create_ts_data():
    from notionhelper import NotionHelper
    import streamlit as st
    # Influenza Data from Notion
    nh = NotionHelper(st.secrets['NOTION_TOKEN'])
    data_id = st.secrets['DATA_ID']
    inf = nh.get_data_source_pages_as_dataframe(data_id)
    
    inf.drop(columns=['Name','notion_page_id'], inplace=True)
    inf.columns = ['inf', 'week']

    inf['week'] = pd.to_datetime(inf['week'], format='mixed', yearfirst=True)
    inf.sort_values(by='week', inplace=True)
    inf.dropna(inplace=True)
    
    # Filtered_df (current FY's appointments / h_data = historic data uploaded.)
    filtered_df = pd.read_csv('/Users/janduplessis/Downloads/access_tracker_filtered_data-5.csv')
    h_data = pd.read_csv('/Users/janduplessis/Library/CloudStorage/OneDrive-NHS/python-data/ACCESS BI/FILTERED_TRAIN_SET.csv')
    
    filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date'], format='%Y-%m-%d')
    h_data['appointment_date'] = pd.to_datetime(h_data['appointment_date'], format='%d-%b-%y')
    
    filtered_df.sort_values(by='appointment_date', inplace=True)
    h_data.sort_values(by='appointment_date', inplace=True)
    
    to_date = filtered_df['appointment_date'].max()
    join_date = pd.Timestamp('2025-04-01')
    from_date = h_data['appointment_date'].min()
    
    filtered_df_new = filtered_df[filtered_df['appointment_date'] >= join_date].copy()
    h_data_new = h_data[h_data['appointment_date'] < join_date].copy()
    
    full_train_apps = pd.concat([h_data_new, filtered_df_new], axis=0)
    full_train = full_train_apps.resample('W', on='appointment_date').size().reset_index(name='apps')
    full_train.columns = ['week', 'apps']
    
    combined = inf.merge(full_train, on='week', how='left')
    combined.dropna(inplace=True)
    
    combined['apps'] = combined['apps'].astype('float32')
    combined['inf'] = combined['inf'].astype('float32')
    combined_ts = TimeSeries.from_dataframe(combined, time_col='week', value_cols=['apps', 'inf'], freq='W')

    
    return combined_ts


def scale_and_split_ts(combined_ts, test_size=64):
    """
    Scales the data and splits it into training and validation sets.
    
    Args:
        df (pd.DataFrame): The combined dataframe.
        combined_ts (TimeSeries): The Darts TimeSeries object created from df.
        test_size (int): The number of periods to hold out for validation.
        
    Returns:
        train_scaled, val_scaled, series_scaled, scaler
    """
    scaler = Scaler()
    
    # 1. Split the data
    # Note: For TimeSeries, we usually split by time rather than random shuffle
    train, val = combined_ts[:-test_size], combined_ts[-test_size:]
    
    # 2. Fit on training data only to avoid data leakage
    scaler.fit(train)
    
    # 3. Transform all sets
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    series_scaled = scaler.transform(combined_ts)
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    train_scaled.plot(label="training")
    val_scaled.plot(label="val")
    plt.title("Appointments + Influenza Index (Scaled)")
    plt.legend()
    plt.show()
    
    return train_scaled, val_scaled, series_scaled, scaler

def find_best_nbeats(train_series, val_series, torch_kwargs=None):
    """
    Performs grid search for NBEATSModel hyperparameters.
    
    Args:
        train_series (TimeSeries): The scaled training series.
        val_series (TimeSeries): The scaled validation series.
        torch_kwargs (dict, optional): Additional lightning/torch arguments.
        
    Returns:
        best_model, best_params, best_score
    """
    
    # Define the parameter grid
    parameters = {
        "input_chunk_length": [104],
        "output_chunk_length": [12],
        "generic_architecture": [False],
        "num_blocks": [5],
        "num_layers": [4],
        "layer_widths": [512],
        "n_epochs": [100],
        "nr_epochs_val_period": [1],
        "batch_size": [64],
        "random_state": [42],
        "force_reset": [True],
        "save_checkpoints": [True]
    }

    # Integrate torch kwargs if provided (ensuring each value is a list for gridsearch)
    if torch_kwargs:
        for k, v in torch_kwargs.items():
            parameters[k] = [v]

    # Run the gridsearch
    # n_jobs=-1 is great for CPU, but if using GPU, 1 is often safer/faster
    best_model, best_params, best_score = NBEATSModel.gridsearch(
        parameters=parameters,
        series=train_series,
        val_series=val_series,
        metric=smape,
        n_jobs=1, 
        verbose=True  # Switched to True to track progress of long runs
    )

    print("-" * 30)
    print(f"Grid Search Complete")
    print(f"Best sMAPE: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    print("-" * 30)

    return best_model, best_params, best_score

# Usage:
# my_torch_args = generate_torch_kwargs()
# model, params, score = find_best_nbeats(train_scaled, val_scaled, my_torch_args)


if __name__ == "__main__":
    print("🚧 Starting Script")
    combined = create_ts_data()
    print("💾 Time Series compiled")
    train_scaled, val_scaled, series_scaled, scaler = scale_and_split_ts(combined, test_size=64)
    print("💾 Time Series scaled")
    print("💾 Starting GridSearch")
    my_torch_args = {
    "pl_trainer_kwargs": {
        "accelerator": "mps", # Use Apple's Metal Performance Shaders
        "devices": 1
    }
}
    best_model, best_params, best_score = find_best_nbeats(train_scaled, val_scaled, my_torch_args)
    
    print(f'🅾️ Best Model: {best_model}')
    print(f'✅ Best Params: {best_params}')
    print(f'💾 Best Score sMAPE: {best_score}')