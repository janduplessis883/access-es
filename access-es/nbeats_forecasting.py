from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import smape, mae, rmse, mape
from darts.dataprocessing.transformers import Scaler, BoxCox
from darts.dataprocessing.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from notionhelper import NotionHelper
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def create_train_data(filtered_df, historic_df, join_date=pd.Timestamp('2025-04-01')):
    """
    Combines influenza data from Notion with historic and current appointment data.
    """
    # 1. Fetch and Clean Influenza Data from Notion
    nh = NotionHelper(st.secrets['NOTION_TOKEN'])
    data_id = st.secrets['DATA_ID']
    inf = nh.get_data_source_pages_as_dataframe(data_id)

    inf.drop(columns=['Name', 'notion_page_id'], inplace=True)
    inf.columns = ['inf', 'week']
    inf['week'] = pd.to_datetime(inf['week'], format='mixed', yearfirst=True)
    inf.sort_values(by='week', inplace=True)
    inf.dropna(inplace=True)

    # 2. Process Appointment Dataframes
    filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date'], format='%Y-%m-%d')
    historic_df['appointment_date'] = pd.to_datetime(historic_df['appointment_date'], format='%d-%b-%y')

    # 3. Split and Recombine based on join_date
    filtered_df_new = filtered_df[filtered_df['appointment_date'] >= join_date].copy()
    historic_df_old = historic_df[historic_df['appointment_date'] < join_date].copy()
    
    full_train_apps = pd.concat([historic_df_old, filtered_df_new], axis=0)
    full_train_apps.sort_values(by='appointment_date', inplace=True)

    # 4. Resample Appointments to Weekly (to match 'inf' frequency)
    # This creates the 'apps_ts' equivalent needed for the merge
    apps_weekly = full_train_apps.resample('W', on='appointment_date').size().reset_index()
    apps_weekly.columns = ['week', 'apps']

    # 5. Merge and Final Clean
    combined = inf.merge(apps_weekly, on='week', how='left')
    combined.dropna(inplace=True)

    # 6. Create Darts TimeSeries
    combined_ts = TimeSeries.from_dataframe(
        combined, 
        time_col='week', 
        value_cols=['apps', 'inf'], 
        freq='W'
    )
    
    # FIX: Explicitly create the Figure and the Axis
    fig, ax = plt.subplots(figsize=(16, 3))
    
    # Tell Darts to plot ONTO the axis we created
    combined_ts.plot(ax=ax)

    return combined_ts, fig



def train_nbeats_model_with_covariates(
    train_ts,
    input_chunk_length=52,
    output_chunk_length=12,
    n_epochs=200,
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    batch_size=64,
    learning_rate=5e-4,
    validation_split=0.2,
    random_state=58,
    accelerator="cpu",
    status_callback=None
):
    """
    Train NBEats model on multivariate TimeSeries with covariates.

    Two-phase training approach:
    - Phase 1: Train with validation split to find optimal stopping point
    - Phase 2: Retrain on full dataset using optimal epochs from Phase 1

    Parameters:
    -----------
    train_ts : TimeSeries
        Multivariate TimeSeries with [appointments, influenza]
    input_chunk_length : int
        Number of past weeks to look at
    output_chunk_length : int
        Number of future weeks to forecast
    n_epochs : int
        Maximum training epochs for Phase 1
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
        update_status("Starting NBEATS model training (two-phase approach)...")

        # 1. Split multivariate series into target and covariate
        target_series = train_ts.univariate_component(0)  # appointments
        covariate_series = train_ts.univariate_component(1)  # influenza

        update_status(f"Prepared data: {len(target_series)} weeks of appointments with influenza covariate")

        # Check minimum data requirements
        min_required = input_chunk_length + output_chunk_length

        if len(target_series) < min_required:
            # Suggest reducing input_chunk_length
            suggested_input = max(output_chunk_length, len(target_series) - output_chunk_length)
            return {
                'success': False,
                'message': f'Insufficient data. With {len(target_series)} weeks, you need input_chunk_length <= {suggested_input} (current: {input_chunk_length}) to support output_chunk_length={output_chunk_length}. Total required: {input_chunk_length + output_chunk_length} weeks.',
                'model': None
            }

        # 2. Calculate validation length
        # Darts requires input_chunk_length + output_chunk_length for valid samples
        min_required_for_training = input_chunk_length + output_chunk_length

        if len(target_series) < min_required_for_training:
            return {
                'success': False,
                'message': f'Insufficient data. Need at least {min_required_for_training} weeks, got {len(target_series)}',
                'model': None
            }

        # For Darts validation, val_series also needs input_chunk_length + output_chunk_length
        # So val_length must be at least min_required_for_training
        min_val_length = min_required_for_training
        max_possible_val = len(target_series) - input_chunk_length

        # Use the larger of the two requirements
        required_val = max(min_val_length, int(len(target_series) * validation_split))
        val_length = min(max_possible_val, required_val)

        train_length = len(target_series) - val_length

        update_status(f"Data allocated: {train_length} weeks training, {val_length} weeks validation")

        # 3. Apply BoxCox transformation to target (for variance stabilization)
        # BoxCox requires strictly positive values, shift if needed
        target_values = target_series.values()
        if np.any(target_values <= 0):
            shift = abs(np.min(target_values)) + 1
            update_status(f"Shifting data by {shift} to make strictly positive for BoxCox")
        else:
            shift = 0

        # Create shifted series if needed
        if shift > 0:
            target_shifted = target_series.map(lambda x: x + shift)
        else:
            target_shifted = target_series

        # Apply BoxCox transformation
        update_status("Applying BoxCox transformation to target series...")
        boxcox_transformer = BoxCox()
        target_boxcox = boxcox_transformer.fit_transform(target_shifted)

        # Then apply MinMax scaling to BoxCox-transformed data
        target_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        target_scaled = target_scaler.fit_transform(target_boxcox)

        # 4. Apply transformations to covariates
        covariate_scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        covariate_scaled = covariate_scaler.fit_transform(covariate_series)

        update_status("Transformations complete")

        # 5. Split for Phase 1 training (train/val)
        # Note: Darts requires val_series to also be >= input_chunk_length + output_chunk_length
        # If we don't have enough data, we'll train without validation and compute metrics later
        train_target = target_scaled[:-val_length]
        train_covariate = covariate_scaled[:-val_length]
        val_target = target_scaled[-val_length:]
        val_covariate = covariate_scaled[-val_length:]

        # Check if val_target is long enough for Darts validation
        min_sample_length = input_chunk_length + output_chunk_length
        has_valid_validation = len(val_target) >= min_sample_length

        if not has_valid_validation:
            update_status(f"Note: Validation set ({len(val_target)} weeks) too small for Darts validation")
            update_status("  Training without validation, will compute metrics post-hoc")

        # 6. Configure Early Stopping for Phase 1
        early_stopper = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode='min',
            verbose=False
        )

        # 7. Initialize NBEATS Model for Phase 1
        update_status("Phase 1: Training to find optimal epochs...")

        model_phase1 = NBEATSModel(
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

        # Phase 1: Train - with or without validation depending on data size
        if has_valid_validation:
            model_phase1.fit(
                series=train_target,
                past_covariates=train_covariate,
                val_series=val_target,
                val_past_covariates=val_covariate,
                verbose=False
            )
        else:
            model_phase1.fit(
                series=train_target,
                past_covariates=train_covariate,
                verbose=False
            )

        # Get number of epochs actually trained
        epochs_trained = early_stopper.stopped_epoch if hasattr(early_stopper, 'stopped_epoch') else n_epochs
        update_status(f"Phase 1 complete: trained for {epochs_trained} epochs")

        # 8. Phase 2: Retrain on FULL dataset using best epoch count
        # Use max of 1 and early_stop_epoch to ensure we train at least a bit
        retrain_epochs = max(1, epochs_trained)
        update_status(f"Phase 2: Retraining on full data ({retrain_epochs} epochs)...")

        # Reset early stopper for Phase 2
        early_stopper_phase2 = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode='min',
            verbose=False
        )

        model_final = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=retrain_epochs,
            batch_size=batch_size,
            optimizer_kwargs={'lr': learning_rate},
            random_state=random_state,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "callbacks": [early_stopper_phase2],
                "enable_progress_bar": False,
                "enable_model_summary": False
            }
        )

        # Train on full data
        model_final.fit(
            series=target_scaled,
            past_covariates=covariate_scaled,
            verbose=False
        )

        update_status("Phase 2 complete: model retrained on full dataset")

        # 9. Calculate validation metrics using Phase 1 model (since it saw val data during training)
        # Generate predictions on validation set
        val_pred_length = min(val_length, output_chunk_length)
        val_target_unscaled = target_series[-val_pred_length:]

        # Generate scaled predictions and inverse transform
        val_pred_scaled = model_phase1.predict(
            n=val_pred_length,
            series=target_scaled[:-val_pred_length],
            past_covariates=covariate_scaled[:-val_pred_length]
        )

        # Inverse transform predictions: BoxCox inverse first, then MinMax inverse
        val_pred_boxcox = target_scaler.inverse_transform(val_pred_scaled)

        if shift > 0:
            val_pred_unscaled = val_pred_boxcox.map(lambda x: x - shift)
        else:
            val_pred_unscaled = val_pred_boxcox

        # Calculate metrics on UNSCALED data
        val_smape = smape(val_target_unscaled, val_pred_unscaled)
        val_mape = mape(val_target_unscaled, val_pred_unscaled)
        val_rmse = rmse(val_target_unscaled, val_pred_unscaled)
        val_mae = mae(val_target_unscaled, val_pred_unscaled)

        update_status(f"Validation Metrics: sMAPE={val_smape:.2f}%, MAPE={val_mape:.2f}%, RMSE={val_rmse:.2f}")

        # 10. Create result dictionary
        result = {
            'success': True,
            'model': model_final,
            'model_phase1': model_phase1,  # Keep Phase 1 model for validation metrics
            'target_scaler': target_scaler,
            'boxcox_transformer': boxcox_transformer,
            'shift_value': shift,
            'covariate_scaler': covariate_scaler,
            'target_series': target_scaled,
            'covariate_series': covariate_scaled,
            'training_weeks': len(target_series),
            'validation_weeks': val_length,
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'epochs_trained': retrain_epochs,
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
        import traceback
        traceback.print_exc()
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
        shift_value = trained_result.get('shift_value', 0)

        # Extend covariate series if future data provided
        if future_covariates_df is not None and len(future_covariates_df) > 0:
            print(f"  Using provided future covariate data ({len(future_covariates_df)} weeks)")

            # Create TimeSeries from future covariates
            future_covariates = TimeSeries.from_dataframe(
                future_covariates_df,
                time_col='week',
                value_cols='influenza',
                freq='W'
            )

            # Scale the future covariates
            future_covariates_scaled = covariate_scaler.transform(future_covariates)

            # Combine historical and future covariates
            extended_covariates = covariate_series.append(future_covariates_scaled)
        else:
            # Use only historical covariates
            extended_covariates = covariate_series

        # Generate forecast
        forecast_scaled = model.predict(
            n=forecast_weeks,
            series=target_series,
            past_covariates=extended_covariates
        )

        # Inverse transform: first inverse MinMax, then inverse BoxCox
        forecast_boxcox = target_scaler.inverse_transform(forecast_scaled)

        # Inverse BoxCox transformation
        if shift_value > 0:
            # First inverse shift
            forecast_unshifted = forecast_boxcox.map(lambda x: x - shift_value)
            # Then inverse BoxCox - need to handle the transformer directly
            forecast = trained_result['boxcox_transformer'].inverse_transform(forecast_unshifted)
        else:
            forecast = trained_result['boxcox_transformer'].inverse_transform(forecast_boxcox)

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
        boxcox_transformer = trained_result['boxcox_transformer']
        shift_value = trained_result.get('shift_value', 0)

        # Helper function to inverse transform predictions with BoxCox
        def inverse_transform_predictions(scaled_series):
            """Inverse transform predictions handling BoxCox and shift"""
            boxcox_series = target_scaler.inverse_transform(scaled_series)
            if shift_value > 0:
                unshifted = boxcox_series.map(lambda x: x - shift_value)
                return boxcox_transformer.inverse_transform(unshifted)
            else:
                return boxcox_transformer.inverse_transform(boxcox_series)

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
        val_pred = inverse_transform_predictions(val_pred_scaled)

        # Generate future forecast
        forecast_scaled = model.predict(
            n=forecast_weeks,
            series=target_series,
            past_covariates=covariate_series
        )
        forecast = inverse_transform_predictions(forecast_scaled)
        
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
