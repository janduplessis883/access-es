from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.metrics import mape, smape, rmse, mae
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st 
from notionhelper import NotionHelper
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter




def process_influenza_data():
    """
    Load and visualize influenza data from Notion database.
    
    Returns:
    --------
    pd.DataFrame
        Influenza data with date and influenza columns
    matplotlib.figure.Figure
        Line plot of influenza over time
    """
    notion_token = st.secrets['NOTION_TOKEN']
    database_id = st.secrets['DATA_ID']
    
    nh = NotionHelper(notion_token)
    inf = nh.get_data_source_pages_as_dataframe(database_id)
    inf['date'] = pd.to_datetime(inf['date'], format='%Y-%m-%d')

    inf = inf.sort_values(by='date').reset_index(drop=True)
    inf.drop(columns=['Name', 'notion_page_id'], inplace=True)

    # Create figure and axes - sns.lineplot returns Axes, not Figure
    fig, ax = plt.subplots(figsize=(16, 2))
    sns.lineplot(data=inf, x='date', y='influenza', ax=ax, color='#515e6b')
    ax.set_title('Influenza Data Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Influenza Level', fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    
    return inf, fig


def process_historic_app_data(combined_df, filtered_df, date_column='appointment_date', app_column='appointment_status'):
    """
    Process uploaded historic training data files and create visualization.
    
    This function:
    1. Filters historic data for 'Finished' appointments before April 2025
    2. Concatenates with current year filtered data (from slider)
    3. Aggregates to weekly frequency for full time period
    4. Creates visualization of appointments over time
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Historic training data (before April 2025)
    filtered_df : pd.DataFrame
        Current year's filtered appointment data (already filtered by slider to END DATE)
    date_column : str
        Name of date column in training data
    app_column : str
        Name of appointment status column in training data
        
    Returns:
    --------
    pd.DataFrame
        Weekly aggregated appointment data with columns ['week', 'appointments']
    matplotlib.figure.Figure
        Line plot showing appointments over time
    """
    
    try:
        # Standardize column names for combined_df
        combined_df.columns = combined_df.columns.str.lower().str.replace(" ", "_").str.strip()

        # Also normalize the app_column and date_column parameters
        app_column = app_column.lower().replace(" ", "_").strip()
        date_column = date_column.lower().replace(" ", "_").strip()

        # Check if app_column exists
        if app_column not in combined_df.columns:
            raise ValueError(f"Column '{app_column}' not found. Available columns: {list(combined_df.columns)}")
        
        # Filter for 'Finished' appointments only
        finished_df = combined_df[combined_df[app_column] == 'Finished'].copy()
        print(f"Filtered to {len(finished_df)} 'Finished' appointments ({len(finished_df)/len(combined_df)*100:.1f}% of total)")
        
        # Check if date_column exists
        if date_column not in finished_df.columns:
            raise ValueError(f"Column '{date_column}' not found. Available columns: {list(finished_df.columns)}")
        
        # Convert date columns to datetime
        finished_df[date_column] = pd.to_datetime(finished_df[date_column])
        finished_df = finished_df.sort_values(by=date_column).reset_index(drop=True)

        # Define cutoff: historic data is before April 1, 2025
        current_start = pd.Timestamp('2025-04-01')
        
        # Filter historic data to before April 2025
        filtered_historic_df = finished_df[finished_df[date_column] < current_start].copy()
        print(f"Historic data (before {current_start.date()}): {len(filtered_historic_df)} appointments")
        
        # Filter current data from April 2025 onwards (already filtered by slider in app.py)
        # Ensure filtered_df has datetime column
        if date_column in filtered_df.columns:
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy[date_column] = pd.to_datetime(filtered_df_copy[date_column])
            current_train_df = filtered_df_copy[filtered_df_copy[date_column] >= current_start].copy()
        else:
            # If filtered_df doesn't have the column, it might be using 'appointment_date'
            filtered_df_copy = filtered_df.copy()
            if 'appointment_date' in filtered_df_copy.columns:
                filtered_df_copy['appointment_date'] = pd.to_datetime(filtered_df_copy['appointment_date'])
                current_train_df = filtered_df_copy[filtered_df_copy['appointment_date'] >= current_start].copy()
                # Rename to match date_column
                current_train_df = current_train_df.rename(columns={'appointment_date': date_column})
            else:
                current_train_df = pd.DataFrame(columns=[date_column])
        
        print(f"Current data (from {current_start.date()} to slider end): {len(current_train_df)} appointments")
        
        # Concatenate historic and current data
        train_apps_df = pd.concat([filtered_historic_df, current_train_df], axis=0, ignore_index=True)
        train_apps_df = train_apps_df.sort_values(by=date_column).reset_index(drop=True)
        
        print(f"Combined training data: {len(train_apps_df)} appointments")
        print(f"Date range: {train_apps_df[date_column].min()} to {train_apps_df[date_column].max()}")

        # Get the training end date from filtered_df (current appointment data from slider)
        # This ensures training data ends at the same date as the current data
        filtered_df_copy = filtered_df.copy()
        if 'appointment_date' in filtered_df_copy.columns:
            filtered_df_copy['appointment_date'] = pd.to_datetime(filtered_df_copy['appointment_date'])
            training_end_date = filtered_df_copy['appointment_date'].max()
        else:
            # Fallback to combined data max date
            training_end_date = train_apps_df[date_column].max()

        print(f"Training end date (from slider): {training_end_date}")
        
        # Aggregate to weekly frequency (week ending Sunday)
        # Set date as index for resampling
        train_apps_df = train_apps_df.set_index(date_column)
        weekly_df = train_apps_df.resample('W-SUN').size().reset_index()
        weekly_df.columns = ['week', 'appointments']
        
        print(f"Aggregated to {len(weekly_df)} weeks")
        print(f"Weekly date range: {weekly_df['week'].min()} to {weekly_df['week'].max()}")
        
        # Fill missing weeks with zeros
        min_week = weekly_df['week'].min()
        max_week = weekly_df['week'].max()
        complete_weeks = pd.date_range(start=min_week, end=max_week, freq='W-SUN')
        complete_df = pd.DataFrame({'week': complete_weeks})
        
        # Merge to fill gaps
        weekly_df = complete_df.merge(weekly_df, on='week', how='left').fillna(0)
        weekly_df['appointments'] = weekly_df['appointments'].astype(int)
        
        print(f"Filled to {len(weekly_df)} complete weeks (gaps filled with 0)")
        
        # Calculate statistics
        mean_apps = weekly_df['appointments'].mean()
        median_apps = weekly_df['appointments'].median()
        total_apps = weekly_df['appointments'].sum()
        
        print(f"\nStatistics:")
        print(f"  Total appointments: {total_apps:,}")
        print(f"  Mean per week: {mean_apps:.1f}")
        print(f"  Median per week: {median_apps:.1f}")
        print(f"  Min per week: {weekly_df['appointments'].min()}")
        print(f"  Max per week: {weekly_df['appointments'].max()}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 3))
        
        # Line plot
        ax.plot(weekly_df['week'], weekly_df['appointments'], color='#a33b54', linewidth=2)
        
        # Add mean line
        ax.axhline(y=mean_apps, color='#ab271f', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_apps:.1f} apps/week', alpha=0.7)
        
        # Styling
        ax.set_title(f'Historic Training Data: Weekly Appointments Over Time\n{len(train_apps_df):,} appointments | {len(weekly_df)} weeks',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Number of Appointments', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)
        ax.legend(fontsize=10)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        plt.tight_layout()
        
        print(f"\n✅ Successfully processed historic training data")
        
        return weekly_df, fig
        
    except Exception as e:
        print(f"\n❌ Error processing historic training data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return empty dataframe and figure on error
        empty_df = pd.DataFrame(columns=['week', 'appointments'])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title('Error Processing Data', fontsize=14, fontweight='bold')
        
        return empty_df, fig

def merge_and_prepare_training_data(appointments_df, influenza_df):
    """
    Merge historic appointments and influenza data, create multivariate TimeSeries.
    
    This function takes preprocessed data from process_historic_app_data() and 
    process_influenza_data(), merges them on weekly timestamps, and creates a
    multivariate Darts TimeSeries ready for training.
    
    Parameters:
    -----------
    appointments_df : pd.DataFrame
        Weekly appointments data with columns ['week', 'appointments']
        Output from process_historic_app_data()
    influenza_df : pd.DataFrame
        Influenza data with columns ['date', 'influenza']
        Output from process_influenza_data()
        
    Returns:
    --------
    TimeSeries
        Multivariate Darts TimeSeries with [appointments, influenza]
    dict
        Metadata about the created dataset
    """
    try:
        print("\n" + "="*60)
        print("MERGING TRAINING DATA")
        print("="*60)
        
        # 1. Prepare appointments data
        df_apps = appointments_df.copy()
        df_apps['week'] = pd.to_datetime(df_apps['week'])

        # Get training end date from appointments data (which comes from slider via process_historic_app_data)
        training_set_end_date = df_apps['week'].max()

        # Trim appointments to training end date (in case historic data extends beyond slider)
        df_apps = df_apps[df_apps['week'] <= training_set_end_date].copy()

        print(f"\nAppointments Data (trimmed to slider end date):")
        print(f"  Weeks: {len(df_apps)}")
        print(f"  Date range: {df_apps['week'].min()} to {df_apps['week'].max()}")
        
        # 2. Prepare influenza data - resample from daily to weekly
        df_flu = influenza_df.copy()
        df_flu['date'] = pd.to_datetime(df_flu['date'])
        
        # Resample to weekly, taking the mean (week ending Sunday to match appointments)
        df_flu = df_flu.set_index('date')
        df_flu_weekly = df_flu.resample('W-SUN')['influenza'].mean().reset_index()
        df_flu_weekly.columns = ['week', 'influenza']
        
        # Trim influenza to training end date
        df_flu_trimmed = df_flu_weekly[df_flu_weekly['week'] <= training_set_end_date].copy()
        
        # Forward fill and back fill any NaN values in influenza data
        df_flu_trimmed['influenza'] = df_flu_trimmed['influenza'].ffill().bfill()
        
        print(f"\nInfluenza Data (trimmed to training end):")
        print(f"  Weeks: {len(df_flu_trimmed)}")
        print(f"  Date range: {df_flu_trimmed['week'].min()} to {df_flu_trimmed['week'].max()}")
        print(f"  First 5 weeks: {df_flu_trimmed['week'].head().tolist()}")
        print(f"  Last 5 weeks: {df_flu_trimmed['week'].tail().tolist()}")

        # 3. Merge appointments and influenza on week
        merged_df = df_apps.merge(df_flu_trimmed, on='week', how='inner')

        print(f"\n🔗 Merged Data:")
        print(f"  Total weeks: {len(merged_df)}")
        if len(merged_df) > 0:
            print(f"  Date range: {merged_df['week'].min()} to {merged_df['week'].max()}")
        else:
            print(f"  Appointments weeks: {df_apps['week'].tolist()}")
            print(f"  Influenza weeks: {df_flu_trimmed['week'].tolist()}")
        
        if len(merged_df) == 0:
            # Provide more diagnostic information
            apps_min = df_apps['week'].min()
            apps_max = df_apps['week'].max()
            flu_min = df_flu_trimmed['week'].min()
            flu_max = df_flu_trimmed['week'].max()
            raise ValueError(
                f"No overlapping weeks between appointments and influenza data.\n"
                f"Appointments date range: {apps_min} to {apps_max}\n"
                f"Influenza date range: {flu_min} to {flu_max}\n\n"
                f"Possible causes:\n"
                f"1. No influenza data in Notion for the appointment date range\n"
                f"2. Influenza data needs to be updated in Notion\n"
                f"3. Check that DATA_ID in secrets matches the influenza database"
            )
        
        # 4. Check for and handle any NaN values
        initial_length = len(merged_df)
        has_nan_mask = merged_df[['appointments', 'influenza']].isna().any(axis=1)
        
        if has_nan_mask.any():
            # Find the first row without NaN
            first_valid_idx = (~has_nan_mask).idxmax() if has_nan_mask.any() else 0
            
            # Drop all rows before the first valid row
            merged_df = merged_df.loc[first_valid_idx:].reset_index(drop=True)
            rows_dropped = initial_length - len(merged_df)
            
            if rows_dropped > 0:
                print(f"\n🧹 Cleaned NaN rows from beginning:")
                print(f"  Dropped {rows_dropped} rows with NaN values")
                print(f"  New date range: {merged_df['week'].min()} to {merged_df['week'].max()}")
                print(f"  New length: {len(merged_df)} weeks")
        
        if len(merged_df) == 0:
            raise ValueError("No valid data remaining after NaN cleaning")
        
        # 5. Create Darts TimeSeries for appointments
        ts_appointments = TimeSeries.from_dataframe(
            merged_df[['week', 'appointments']], 
            time_col='week', 
            value_cols='appointments', 
            freq='W'
        )
        
        # 6. Create Darts TimeSeries for influenza
        ts_influenza = TimeSeries.from_dataframe(
            merged_df[['week', 'influenza']], 
            time_col='week', 
            value_cols='influenza', 
            freq='W'
        )
        
        # 7. Stack into multivariate series [appointments, influenza]
        train_ts = ts_appointments.stack(ts_influenza)
        
        # 8. Create metadata
        metadata = {
            'success': True,
            'total_weeks': len(train_ts),
            'start_date': train_ts.start_time(),
            'end_date': train_ts.end_time(),
            'training_weeks': len(merged_df),
            'current_weeks': 0,
            'components': train_ts.components.tolist(),
            'message': f'Successfully merged {len(train_ts)} weeks of data',
            'appointments_series': ts_appointments,
            'trimmed_to_slider_date': True
        }
        
        print(f"\n✅ Training Dataset Created:")
        print(f"  Total weeks: {len(train_ts)}")
        print(f"  Date range: {train_ts.start_time()} to {train_ts.end_time()}")
        print(f"  Components: {train_ts.components.tolist()}")
        print("="*60 + "\n")
        
        return train_ts, metadata
        
    except Exception as e:
        print(f"\n❌ Error merging training data: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        metadata = {
            'success': False,
            'message': f'Error: {str(e)}',
            'total_weeks': 0
        }
        return None, metadata
    


def train_nbeats_model_with_covariates(
    train_ts,
    input_chunk_length=52,
    output_chunk_length=12,
    n_epochs=100,
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    batch_size=32,
    learning_rate=5e-4,
    validation_split=0.2,
    random_state=58,
    accelerator="cpu",
    status_callback=None
):
    """
    Train NBEats model on multivariate TimeSeries with covariates.

    Parameters:
    -----------
    train_ts : TimeSeries
        Multivariate TimeSeries with [appointments, influenza]
    input_chunk_length : int
        Number of past weeks to look at
    output_chunk_length : int
        Number of future weeks to forecast
    n_epochs : int
        Maximum training epochs
    validation_split : float
        Fraction of data reserved for validation
    status_callback : callable, optional
        Function to call with status updates (receives status message string)

    Returns:
    --------
    dict
        Training results with model, scalers, and metadata
    """

    def update_status(msg):
        if status_callback:
            status_callback(msg)
        print(f"  📍 {msg}")

    try:
        update_status("Starting NBEATS model training...")

        # 1. Split multivariate series into target and covariate
        target_series = train_ts.univariate_component(0)  # appointments
        covariate_series = train_ts.univariate_component(1)  # influenza

        update_status(f"Prepared data: {len(target_series)} weeks of appointments with influenza covariate")

        # Check minimum data requirements
        min_required = input_chunk_length + output_chunk_length
        if len(target_series) < min_required:
            return {
                'success': False,
                'message': f'Insufficient data. Need at least {min_required} weeks, got {len(target_series)}',
                'model': None
            }

        # 2. Calculate validation length
        val_length = max(output_chunk_length, int(len(target_series) * validation_split))
        train_length = len(target_series) - val_length

        update_status(f"Data allocated: {train_length} weeks training, {val_length} weeks validation")

        # 3. Apply transformations to target
        update_status("Applying Min-Max scaling to data...")
        target_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        target_scaled = target_scaler.fit_transform(target_series)

        # 4. Apply transformations to covariates
        covariate_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        covariate_scaled = covariate_scaler.fit_transform(covariate_series)

        update_status("Scaling complete")

        # 5. Configure Early Stopping
        early_stopper = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode='min',
        )

        # 6. Initialize NBEats Model
        update_status("Initializing NBEATS neural network architecture...")

        model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={'lr': learning_rate},
            random_state=random_state,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "callbacks": [early_stopper],
                "enable_progress_bar": False,
                "enable_model_summary": False
            }
        )

        # 7. Train the model
        update_status(f"Training model (up to {n_epochs} epochs, early stopping enabled)...")

        model.fit(
            series=target_scaled,
            past_covariates=covariate_scaled,
            verbose=False
        )

        update_status("Training complete! Calculating validation metrics...")
        
        # Limit validation prediction to output_chunk_length (we don't have future covariates beyond that)
        val_pred_length = min(val_length, output_chunk_length)
        
        # Get validation data - UNSCALED (last val_pred_length weeks)
        val_target_unscaled = target_series[-val_pred_length:]
        
        # Also get the scaled version for prediction
        val_target_scaled = target_scaled[-val_pred_length:]
        
        # Generate SCALED predictions on validation set (limited to output_chunk_length)
        val_pred_scaled = model.predict(
            n=val_pred_length,
            series=target_scaled[:-val_pred_length],
            past_covariates=covariate_scaled[:-val_pred_length]
        )
        
        # IMPORTANT: Inverse transform predictions to UNSCALED for metrics
        val_pred_unscaled = target_scaler.inverse_transform(val_pred_scaled)
        
        # Calculate metrics on UNSCALED data (actual vs predicted appointments)
        val_smape = smape(val_target_unscaled, val_pred_unscaled)
        val_mape = mape(val_target_unscaled, val_pred_unscaled)
        val_rmse = rmse(val_target_unscaled, val_pred_unscaled)
        val_mae = mae(val_target_unscaled, val_pred_unscaled)

        update_status(f"Validation: sMAPE={val_smape:.2f}%, MAPE={val_mape:.2f}%, RMSE={val_rmse:.2f}")

        # 9. Create result dictionary
        result = {
            'success': True,
            'model': model,
            'target_scaler': target_scaler,
            'covariate_scaler': covariate_scaler,
            'target_series': target_scaled,
            'covariate_series': covariate_scaled,
            'training_weeks': len(target_series),
            'validation_weeks': val_length,
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'message': f'Model trained successfully on {len(target_series)} weeks',
            'metrics': {
                'smape': float(val_smape),
                'mape': float(val_mape),
                'rmse': float(val_rmse),
                'mae': float(val_mae)
            }
        }

        update_status("Model training complete and packaged!")
        return result

    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        if status_callback:
            status_callback(f"Error: {error_msg}")
        print(f"\n❌ {error_msg}\n")
        return {
            'success': False,
            'message': error_msg,
            'model': None
        }


def forecast_nbeats_model(
    trained_result,
    forecast_weeks=12,
    future_covariates_df=None
):
    """
    Generate forecasts using trained NBEats model.
    
    Parameters:
    -----------
    trained_result : dict
        Dictionary from train_nbeats_model_with_covariates()
    forecast_weeks : int
        Number of weeks to forecast
    future_covariates_df : pd.DataFrame (optional)
        Future influenza data with columns ['week', 'influenza']
        
    Returns:
    --------
    pd.DataFrame
        Forecast with columns ['week', 'forecasted_appointments']
    """
    
    if not trained_result or not trained_result.get('success'):
        print("❌ No trained model available")
        return None
    
    try:
        print(f"\n🔮 Generating {forecast_weeks}-week forecast...")
        
        model = trained_result['model']
        target_scaler = trained_result['target_scaler']
        covariate_scaler = trained_result['covariate_scaler']
        target_series = trained_result['target_series']
        covariate_series = trained_result['covariate_series']
        
        # Extend covariate series if future data provided
        if future_covariates_df is not None:
            print(f"  Using provided future covariate data")
            # TODO: Implement future covariate extension
        
        # Generate forecast
        forecast_scaled = model.predict(
            n=forecast_weeks,
            series=target_series,
            past_covariates=covariate_series
        )
        
        # Inverse transform
        forecast = target_scaler.inverse_transform(forecast_scaled)
        
        # Convert to DataFrame using values() and time_index
        forecast_df = pd.DataFrame({
            'week': forecast.time_index,
            'forecasted_appointments': forecast.values().flatten()
        })
        
        # Round and clip values
        forecast_df['forecasted_appointments'] = forecast_df['forecasted_appointments'].round(0).clip(lower=0).astype(int)
        
        print(f"✅ Forecast generated successfully")
        print(f"  Date range: {forecast_df['week'].min()} to {forecast_df['week'].max()}")
        print(f"  Mean forecast: {forecast_df['forecasted_appointments'].mean():.0f} appointments/week\n")
        
        return forecast_df
        
    except Exception as e:
        print(f"❌ Forecasting error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return None


def plot_validation_and_forecast(
    trained_result,
    training_metadata=None,
    validation_weeks=12,
    forecast_weeks=12
):
    """
    Create a plot showing training data, validation period, and forecast.
    
    Parameters:
    -----------
    trained_result : dict
        Dictionary from train_nbeats_model_with_covariates()
    training_metadata : dict
        Metadata from create_training_dataset() containing full appointments series
    validation_weeks : int
        Number of weeks to use for validation visualization
    forecast_weeks : int
        Number of weeks to forecast and visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot with training, validation, and forecast
    """
    
    if not trained_result or not trained_result.get('success'):
        return None
    
    try:
        model = trained_result['model']
        target_scaler = trained_result['target_scaler']
        covariate_scaler = trained_result['covariate_scaler']
        target_series = trained_result['target_series']
        covariate_series = trained_result['covariate_series']
        
        # Get full appointments series from metadata (includes both historical and current)
        if training_metadata and 'appointments_series' in training_metadata:
            full_appointments = training_metadata['appointments_series']
            print(f"Using full combined appointments data: {len(full_appointments)} weeks")
        else:
            # Fallback to using target_series
            full_appointments = target_scaler.inverse_transform(target_series)
            print(f"Using model target series: {len(full_appointments)} weeks")
        
        # Split into train and validation using full data
        train_series = full_appointments[:-validation_weeks]
        val_series = full_appointments[-validation_weeks:]
        
        # Generate validation predictions
        val_pred_scaled = model.predict(
            n=validation_weeks,
            series=target_series[:-validation_weeks],
            past_covariates=covariate_series[:-validation_weeks]
        )
        val_pred = target_scaler.inverse_transform(val_pred_scaled)
        
        # Generate future forecast
        forecast_scaled = model.predict(
            n=forecast_weeks,
            series=target_series,
            past_covariates=covariate_series
        )
        forecast = target_scaler.inverse_transform(forecast_scaled)
        
        # Convert to pandas for plotting using values() and time_index
        train_df = pd.DataFrame({
            'week': train_series.time_index,
            'appointments': train_series.values().flatten()
        })
        
        val_actual_df = pd.DataFrame({
            'week': val_series.time_index,
            'appointments': val_series.values().flatten()
        })
        
        val_pred_df = pd.DataFrame({
            'week': val_pred.time_index,
            'appointments': val_pred.values().flatten()
        })
        
        forecast_df = pd.DataFrame({
            'week': forecast.time_index,
            'appointments': forecast.values().flatten()
        })
        
        # Create plot
        fig = go.Figure()
        
        # Training data
        fig.add_trace(go.Scatter(
            x=train_df['week'],
            y=train_df['appointments'],
            mode='lines',
            name='Training Data',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Validation actual
        fig.add_trace(go.Scatter(
            x=val_actual_df['week'],
            y=val_actual_df['appointments'],
            mode='lines+markers',
            name='Validation (Actual)',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6)
        ))
        
        # Validation predicted
        fig.add_trace(go.Scatter(
            x=val_pred_df['week'],
            y=val_pred_df['appointments'],
            mode='lines+markers',
            name='Validation (Predicted)',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='x')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['week'],
            y=forecast_df['appointments'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#d62728', width=2.5, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Update layout
        fig.update_layout(
            title='NBEats Model: Training, Validation & Forecast',
            xaxis_title='Week',
            yaxis_title='Appointments',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"❌ Plotting error: {str(e)}")
        return None
    
    try:
        print(f"\n🔮 Generating {forecast_weeks}-week forecast...")
        
        model = trained_result['model']
        target_scaler = trained_result['target_scaler']
        covariate_scaler = trained_result['covariate_scaler']
        target_series = trained_result['target_series']
        covariate_series = trained_result['covariate_series']
        
        # Extend covariate series if future data provided
        if future_covariates_df is not None:
            print(f"  Using provided future covariate data")
            # TODO: Implement future covariate extension
        
        # Generate forecast
        forecast_scaled = model.predict(
            n=forecast_weeks,
            series=target_series,
            past_covariates=covariate_series
        )
        
        # Inverse transform
        forecast = target_scaler.inverse_transform(forecast_scaled)
        
        # Convert to DataFrame
        forecast_df = forecast.pd_dataframe()
        forecast_df = forecast_df.reset_index()
        forecast_df.columns = ['week', 'forecasted_appointments']
        forecast_df['forecasted_appointments'] = forecast_df['forecasted_appointments'].round(0).clip(lower=0)
        
        print(f"✅ Forecast generated successfully")
        print(f"  Date range: {forecast_df['week'].min()} to {forecast_df['week'].max()}")
        print(f"  Mean forecast: {forecast_df['forecasted_appointments'].mean():.0f} appointments/week\n")
        
        return forecast_df
        
    except Exception as e:
        print(f"❌ Forecasting error: {str(e)}\n")
        return None
