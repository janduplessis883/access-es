from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_training_dataset(
    training_df,
    weekly_agg_df,
    date_column='Appointment date',
    app_column='used_apps',
    influenza_csv_path='data/influenza.csv',
    training_cutoff_date='2025-03-30',
    current_start_date='2025-04-06'
):
    """
    Create a combined training dataset from historical training data and current appointments.
    
    Parameters:
    -----------
    training_df : pd.DataFrame
        Historical training data (user uploaded)
    weekly_agg_df : pd.DataFrame
        Current weekly aggregated data from app
    date_column : str
        Name of date column in training_df
    app_column : str
        Name of appointments column in training_df
    influenza_csv_path : str
        Path to influenza covariate data
    training_cutoff_date : str
        End date for training data (week ending, e.g., '2025-03-30')
    current_start_date : str
        Start date for current data (week ending, e.g., '2025-04-06')
    
    Returns:
    --------
    TimeSeries
        Multivariate Darts TimeSeries with appointments and influenza
    dict
        Metadata about the created dataset
    """
    try:
        # 1. Load and prepare first appointment dataset (training data)
        df_appts_1 = training_df.copy()
        
        # FIRST: Filter to only include 'Finished' appointments
        if app_column in df_appts_1.columns:
            # Filter where appointment status = 'Finished'
            df_appts_1 = df_appts_1[df_appts_1[app_column] == 'Finished']
            print(f"Filtered to {len(df_appts_1)} 'Finished' appointments from training data")
        else:
            raise ValueError(f"Column '{app_column}' (Appointment Status) not found in training data")
        
        # Ensure date column is datetime
        if date_column in df_appts_1.columns:
            df_appts_1[date_column] = pd.to_datetime(df_appts_1[date_column])
        else:
            raise ValueError(f"Column '{date_column}' not found in training data")
        
        # TEMPORARY: Trim to specific date range (3 Oct 2023 - 30 March 2025)
        start_cutoff = pd.to_datetime('2023-10-01')
        end_cutoff = pd.to_datetime(training_cutoff_date)
        
        df_appts_1 = df_appts_1[
            (df_appts_1[date_column] >= start_cutoff) & 
            (df_appts_1[date_column] <= end_cutoff)
        ]
        print(f"Trimmed to {len(df_appts_1)} appointments between {start_cutoff.date()} and {end_cutoff.date()}")
        
        # Set date as index for resampling
        df_appts_1 = df_appts_1.set_index(date_column)
        
        # Aggregate to weekly using resample().size()
        df_appts_1_weekly = df_appts_1.resample('W').size().reset_index()
        df_appts_1_weekly.columns = ['week', 'appointments']
        print(f"Aggregated to {len(df_appts_1_weekly)} weeks of data")
        
        # Create complete weekly range and fill missing values
        min_week = df_appts_1_weekly['week'].min()
        max_week = df_appts_1_weekly['week'].max()
        complete_weeks = pd.date_range(start=min_week, end=max_week, freq='W')
        complete_df = pd.DataFrame({'week': complete_weeks})
        df_appts_1_weekly = complete_df.merge(df_appts_1_weekly, on='week', how='left').fillna(0)
        print(f"Filled to {len(df_appts_1_weekly)} complete weeks (missing weeks filled with 0)")
        
        # 2. Load second appointment dataset (current data from app)
        df_appts_2 = weekly_agg_df.copy()
        
        # Filter to start from specified date (week ending 6 April 2025)
        start_date = pd.to_datetime(current_start_date)
        if 'week' in df_appts_2.columns:
            df_appts_2['week'] = pd.to_datetime(df_appts_2['week'])
            df_appts_2 = df_appts_2[df_appts_2['week'] >= start_date]
            df_appts_2 = df_appts_2[['week', 'total_appointments']].copy()
            df_appts_2.columns = ['week', 'appointments']
            
            # Fill missing weeks in current data too
            if len(df_appts_2) > 0:
                min_week_2 = df_appts_2['week'].min()
                max_week_2 = df_appts_2['week'].max()
                complete_weeks_2 = pd.date_range(start=min_week_2, end=max_week_2, freq='W')
                complete_df_2 = pd.DataFrame({'week': complete_weeks_2})
                df_appts_2 = complete_df_2.merge(df_appts_2, on='week', how='left').fillna(0)
                print(f"Current data: {len(df_appts_2)} complete weeks")
        else:
            raise ValueError("'week' column not found in weekly_agg_df")
        
        # 3. Convert to Darts TimeSeries (no need for fill_missing_dates now)
        ts_appts_1 = TimeSeries.from_dataframe(
            df_appts_1_weekly, 
            time_col='week', 
            value_cols='appointments', 
            freq='W'
        )
        
        ts_appts_2 = TimeSeries.from_dataframe(
            df_appts_2, 
            time_col='week', 
            value_cols='appointments', 
            freq='W'
        )
        
        # 4. Merge them chronologically (use ignore_time_axis=True to handle gap)
        ts_appointments = ts_appts_1.concatenate(ts_appts_2, ignore_time_axis=True)
        print(f"Concatenated training and current data: {len(ts_appointments)} total weeks")
        
        # 5. Load Influenza data
        if not os.path.exists(influenza_csv_path):
            raise FileNotFoundError(f"Influenza data not found at {influenza_csv_path}")
        
        df_flu = pd.read_csv(influenza_csv_path)
        df_flu['date'] = pd.to_datetime(df_flu['date'])
        
        # Aggregate to weekly if needed
        df_flu['week'] = df_flu['date'].dt.to_period('W').dt.start_time
        df_flu_weekly = df_flu.groupby('week')['influenza'].mean().reset_index()
        
        # Fill missing weeks in influenza data too
        min_week_flu = df_flu_weekly['week'].min()
        max_week_flu = df_flu_weekly['week'].max()
        complete_weeks_flu = pd.date_range(start=min_week_flu, end=max_week_flu, freq='W')
        complete_df_flu = pd.DataFrame({'week': complete_weeks_flu})
        df_flu_weekly = complete_df_flu.merge(df_flu_weekly, on='week', how='left')
        # Forward fill influenza values (use last known value for gaps)
        df_flu_weekly['influenza'] = df_flu_weekly['influenza'].ffill().bfill()
        print(f"Influenza data: {len(df_flu_weekly)} complete weeks")
        
        ts_flu_full = TimeSeries.from_dataframe(
            df_flu_weekly, 
            time_col='week', 
            value_cols='influenza', 
            freq='W'
        )
        
        # 6. Slice the influenza data to match the appointment start/end dates
        ts_flu_clipped = ts_flu_full.slice_intersect(ts_appointments)
        
        # 7. Combine into a multivariate series
        # This creates a single object where each timestamp has 2 components: [appointments, influenza]
        train_ts = ts_appointments.stack(ts_flu_clipped)
        
        # 8. Create metadata (store original appointment series for plotting)
        metadata = {
            'success': True,
            'total_weeks': len(train_ts),
            'start_date': train_ts.start_time(),
            'end_date': train_ts.end_time(),
            'training_weeks': len(ts_appts_1),
            'current_weeks': len(ts_appts_2),
            'components': train_ts.components.tolist(),
            'message': f'Successfully created training dataset with {len(train_ts)} weeks',
            'appointments_series': ts_appointments  # Store complete appointments for plotting
        }
        
        print(f"✓ Training dataset created: {len(train_ts)} weeks")
        print(f"  - Historical training: {len(ts_appts_1)} weeks (up to {end_cutoff.date()})")
        print(f"  - Current data: {len(ts_appts_2)} weeks (from {start_date.date()})")
        print(f"  - Date range: {train_ts.start_time()} to {train_ts.end_time()}")
        print(f"  - Components: {train_ts.components.tolist()}")
        
        return train_ts, metadata
        
    except Exception as e:
        print(f"Error creating training dataset: {str(e)}")
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
    accelerator="cpu"
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
        
    Returns:
    --------
    dict
        Training results with model, scalers, and metadata
    """
    
    try:
        print("\n" + "="*60)
        print("NBEATS MODEL TRAINING")
        print("="*60)
        
        # 1. Split multivariate series into target and covariate
        target_series = train_ts.univariate_component(0)  # appointments
        covariate_series = train_ts.univariate_component(1)  # influenza
        
        print(f"\n📊 Data Split:")
        print(f"  Target (appointments): {len(target_series)} weeks")
        print(f"  Covariate (influenza): {len(covariate_series)} weeks")
        
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
        
        print(f"\n📈 Data Allocation:")
        print(f"  Training: {train_length} weeks")
        print(f"  Validation: {val_length} weeks")
        print(f"  Input chunk: {input_chunk_length} weeks")
        print(f"  Output chunk: {output_chunk_length} weeks")
        
        # 3. Apply transformations to target
        print(f"\n🔄 Applying Min-Max Scaling...")
        target_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        target_scaled = target_scaler.fit_transform(target_series)
        
        # 4. Apply transformations to covariates
        covariate_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        covariate_scaled = covariate_scaler.fit_transform(covariate_series)
        
        print(f"  ✓ Target scaled to range [0, 1]")
        print(f"  ✓ Covariate scaled to range [0, 1]")
        
        # 5. Configure Early Stopping
        early_stopper = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode='min',
        )
        
        # 6. Initialize NBEats Model
        print(f"\n🧠 Initializing NBEats Model:")
        print(f"  Architecture: Generic")
        print(f"  Stacks: {num_stacks}")
        print(f"  Blocks per stack: {num_blocks}")
        print(f"  Layers per block: {num_layers}")
        print(f"  Layer width: {layer_widths}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
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
                "enable_progress_bar": True,
                "enable_model_summary": True
            }
        )
        
        # 7. Train the model
        print(f"\n🚀 Training Model (max {n_epochs} epochs)...")
        print(f"   Early stopping enabled (patience=15)")
        print("-" * 60)
        
        model.fit(
            series=target_scaled,
            past_covariates=covariate_scaled,
            verbose=True
        )
        
        print("-" * 60)
        print(f"✅ Training Complete!")
        
        # 8. Create result dictionary
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
            'message': f'Model trained successfully on {len(target_series)} weeks'
        }
        
        print(f"\n📦 Model Package Created:")
        print(f"  ✓ Trained NBEats model")
        print(f"  ✓ Target scaler")
        print(f"  ✓ Covariate scaler")
        print(f"  ✓ Scaled time series")
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Training Error: {str(e)}\n")
        return {
            'success': False,
            'message': f'Training error: {str(e)}',
            'model': None
        }


def forecast_nbeats_model(
    trained_result,
    forecast_weeks=4,
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
