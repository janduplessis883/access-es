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

def run_nbeats_forecast(train_scaled, val_scaled, series_scaled, scaler, user_params):
    """
    Trains an NBEATS model with user-selected parameters and forecasts the validation period.
    
    Args:
        train_scaled (TimeSeries): Scaled training data.
        val_scaled (TimeSeries): Scaled validation data (for evaluation).
        series_scaled (TimeSeries): Full scaled series (optional, if you want to retrain on full data later).
        scaler (Scaler): Fitted scaler for inverse transformation.
        user_params (dict): Dictionary containing model hyperparameters.
        
    Returns:
        forecast (TimeSeries): The predicted series (inverse transformed).
        model (NBEATSModel): The trained model object.
    """
    
    print(f"⚙️ Training NBEATS with params: {user_params}")

    # 1. Initialize Model
    # We map the user_params dict to the model arguments
    model = NBEATSModel(
        input_chunk_length=user_params['input_chunk_length'],
        output_chunk_length=user_params['output_chunk_length'],
        generic_architecture=user_params['generic_architecture'],
        num_blocks=user_params['num_blocks'],
        num_layers=user_params['num_layers'],
        layer_widths=user_params['layer_widths'],
        n_epochs=user_params['n_epochs'],
        batch_size=user_params['batch_size'],
        nr_epochs_val_period=1,
        random_state=42,
        # Adjust accelerator based on your hardware (cpu, gpu, auto)
        pl_trainer_kwargs={"accelerator": "cpu"} 
    )
    
    # 2. Fit the model
    # We use val_scaled for validation loss monitoring
    model.fit(series=train_scaled, val_series=val_scaled, verbose=True)
    
    # 3. Predict
    # Forecast for the length of the validation set
    pred_len = len(val_scaled)
    pred_scaled = model.predict(n=pred_len)
    
    # 4. Inverse Transform (bring back to original units)
    # We must inverse transform both the prediction and the validation set for plotting
    pred = scaler.inverse_transform(pred_scaled)
    val_original = scaler.inverse_transform(val_scaled)
    train_original = scaler.inverse_transform(train_scaled)

    # 5. Calculate Error
    error_score = smape(val_original, pred)
    print(f"✅ Forecast Complete. sMAPE: {error_score:.2f}%")
    
    # 6. Plotting
    plt.figure(figsize=(12, 6))
    train_original['apps'].plot(label="Training Data")
    val_original['apps'].plot(label="Actual Validation Data")
    pred['apps'].plot(label=f"NBEATS Forecast (sMAPE: {error_score:.2f}%)", linestyle='--')
    plt.title("NBEATS Forecast vs Actuals")
    plt.legend()
    plt.show()
    
    return pred, model

if __name__ == "__main__":
    print("🚧 Starting Script")
    
    # 1. Load Data
    try:
        combined = create_ts_data()
        print("💾 Time Series compiled")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data for testing if files aren't found
        print("Creating dummy data for demonstration...")
        dates = pd.date_range('2023-01-01', periods=100, freq='W')
        values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100) + 10
        inf_values = np.cos(np.linspace(0, 10, 100)) + 5
        df = pd.DataFrame({'week': dates, 'apps': values, 'inf': inf_values})
        combined = TimeSeries.from_dataframe(df, time_col='week', value_cols=['apps', 'inf'])

    # 2. Scale and Split
    train_scaled, val_scaled, series_scaled, scaler = scale_and_split_ts(combined, test_size=76)
    print("💾 Time Series scaled")
    
    # 3. Define User Parameters (These would come from Streamlit widgets in the real app)
    # Example: selected_params = st.session_state['user_params']
    selected_params = {
        "input_chunk_length": 64,
        "output_chunk_length": 12,
        "generic_architecture": True,
        "num_blocks": 4,
        "num_layers": 4,
        "layer_widths": 512,
        "n_epochs": 100,          # kept low for testing speed
        "batch_size": 800
    }

    # 4. Run Forecast
    forecast_series, trained_model = run_nbeats_forecast(
        train_scaled, 
        val_scaled, 
        series_scaled, 
        scaler, 
        selected_params
    )