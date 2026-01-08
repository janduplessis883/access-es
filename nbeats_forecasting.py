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
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.lineplot(data=inf, x='date', y='influenza', ax=ax, color='black')
    ax.set_title('Influenza Data Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Influenza Level', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return inf, fig


def process_historic_app_data(training_files, date_column='Appointment date', app_column='Appointment status'):
    """
    Process uploaded historic training data files and create visualization.
    
    This function:
    1. Combines multiple uploaded CSV files
    2. Filters for 'Finished' appointments only
    3. Aggregates to weekly frequency for full time period
    4. Creates visualization of appointments over time
    
    Parameters:
    -----------
    training_files : list
        List of uploaded file objects from st.file_uploader
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
        # 1. Load and combine all training files
        training_dfs = []
        for idx, file in enumerate(training_files):
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file)
            print(f"Loaded file {idx+1} ({file.name}): {len(df)} rows")
            training_dfs.append(df)
        
        if not training_dfs:
            raise ValueError("No training files provided")
        
        # Combine all files
        combined_df = pd.concat(training_dfs, ignore_index=True)
        print(f"Combined {len(training_dfs)} files: {len(combined_df)} total rows")
        
        # 2. Filter for 'Finished' appointments only
        if app_column not in combined_df.columns:
            raise ValueError(f"Column '{app_column}' not found. Available columns: {list(combined_df.columns)}")
        
        finished_df = combined_df[combined_df[app_column] == 'Finished'].copy()
        print(f"Filtered to {len(finished_df)} 'Finished' appointments ({len(finished_df)/len(combined_df)*100:.1f}% of total)")
        
        # 3. Convert date column to datetime
        if date_column not in finished_df.columns:
            raise ValueError(f"Column '{date_column}' not found. Available columns: {list(finished_df.columns)}")
        
        finished_df[date_column] = pd.to_datetime(finished_df[date_column])
        
        # 4. Set date as index for resampling
        finished_df = finished_df.set_index(date_column)
        
        # 5. Aggregate to weekly frequency (full time period)
        weekly_df = finished_df.resample('W').size().reset_index()
        weekly_df.columns = ['week', 'appointments']
        print(f"Aggregated to {len(weekly_df)} weeks")
        print(f"Date range: {weekly_df['week'].min()} to {weekly_df['week'].max()}")
        
        # 6. Fill missing weeks with zeros
        min_week = weekly_df['week'].min()
        max_week = weekly_df['week'].max()
        complete_weeks = pd.date_range(start=min_week, end=max_week, freq='W')
        complete_df = pd.DataFrame({'week': complete_weeks})
        weekly_df = complete_df.merge(weekly_df, on='week', how='left').fillna(0)
        print(f"Filled to {len(weekly_df)} complete weeks (gaps filled with 0)")
        
        # 7. Add summary statistics
        weekly_df['appointments'] = weekly_df['appointments'].astype(int)
        mean_apps = weekly_df['appointments'].mean()
        median_apps = weekly_df['appointments'].median()
        total_apps = weekly_df['appointments'].sum()
        
        print(f"\nStatistics:")
        print(f"  Total appointments: {total_apps:,}")
        print(f"  Mean per week: {mean_apps:.1f}")
        print(f"  Median per week: {median_apps:.1f}")
        print(f"  Min per week: {weekly_df['appointments'].min()}")
        print(f"  Max per week: {weekly_df['appointments'].max()}")
        
        # 8. Create visualization
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Main line plot
        sns.barplot(data=weekly_df, x='week', y='appointments', ax=ax, linewidth=2, color='#a33b54')
        
        # Add mean line
        ax.axhline(y=mean_apps, color='#ab271f', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_apps:.1f} apps/week', alpha=0.7)
        
        # Styling
        ax.set_title(f'Historic Training Data: Weekly Appointments Over Time\n'
                    f'{len(training_dfs)} files | {len(combined_df):,} total rows | '
                    f'{len(finished_df):,} finished appointments | {len(weekly_df)} weeks',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Number of Appointments', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        plt.tight_layout()
        
        print(f"\n✅ Successfully processed historic training data")
        
        return weekly_df, fig
        
    except Exception as e:
        print(f"\n❌ Error processing historic training data: {str(e)}")
        
        # Return empty dataframe and figure on error
        empty_df = pd.DataFrame(columns=['week', 'appointments'])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title('Error Processing Data', fontsize=14, fontweight='bold')
        
        return empty_df, fig

def plot_merged_training_data(appointments_df, influenza_df):
    """
    Create visualization of merged appointments and influenza data.
    
    Parameters:
    -----------
    appointments_df : pd.DataFrame
        Weekly appointments data with columns ['week', 'appointments']
    influenza_df : pd.DataFrame
        Influenza data with columns ['date', 'influenza']
        
    Returns:
    --------
    matplotlib.figure.Figure
        Dual-axis plot showing appointments and influenza over time
    pd.DataFrame
        Merged dataframe with both appointments and influenza
    """
    try:
        # Prepare appointments data
        df_apps = appointments_df.copy()
        df_apps['week'] = pd.to_datetime(df_apps['week'])
        
        # Prepare influenza data - aggregate to weekly
        df_flu = influenza_df.copy()
        df_flu['date'] = pd.to_datetime(df_flu['date'])
        df_flu = df_flu.set_index('date')
        df_flu_weekly = df_flu.resample('W')['influenza'].mean().reset_index()
        df_flu_weekly.columns = ['week', 'influenza']
        
        # Fill missing weeks in influenza
        min_week_flu = df_flu_weekly['week'].min()
        max_week_flu = df_flu_weekly['week'].max()
        complete_weeks_flu = pd.date_range(start=min_week_flu, end=max_week_flu, freq='W')
        complete_df_flu = pd.DataFrame({'week': complete_weeks_flu})
        df_flu_weekly = complete_df_flu.merge(df_flu_weekly, on='week', how='left')
        df_flu_weekly['influenza'] = df_flu_weekly['influenza'].ffill().bfill()
        
        # Merge appointments and influenza on week
        merged_df = df_apps.merge(df_flu_weekly, on='week', how='inner')
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(16, 4))
        
        # Plot appointments on primary axis
        color1 = '#1f77b4'
        ax1.set_xlabel('Week', fontsize=12)
        ax1.set_ylabel('Appointments', color=color1, fontsize=12)
        ax1.plot(merged_df['week'], merged_df['appointments'], 
                color=color1, linewidth=2, label='Appointments')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Create secondary axis for influenza
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('Influenza Level', color=color2, fontsize=12)
        ax2.plot(merged_df['week'], merged_df['influenza'], 
                color=color2, linewidth=2, linestyle='-', label='Influenza')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Title and legend
        plt.title(f'Merged Training Data: Appointments & Influenza Over Time\n'
                 f'{len(merged_df)} weeks | {merged_df["week"].min().date()} to {merged_df["week"].max().date()}',
                 fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        print(f"\n✅ Created merged visualization: {len(merged_df)} weeks")
        
        return fig, merged_df
        
    except Exception as e:
        print(f"\n❌ Error creating merged visualization: {str(e)}")
        
        # Return empty figure on error
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title('Error Creating Merged Plot', fontsize=14, fontweight='bold')
        
        empty_df = pd.DataFrame(columns=['week', 'appointments', 'influenza'])
        return fig, empty_df


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
        print(f"\n📊 Appointments Data:")
        print(f"  Weeks: {len(df_apps)}")
        print(f"  Date range: {df_apps['week'].min()} to {df_apps['week'].max()}")
        
        # 2. Prepare influenza data - aggregate to weekly
        df_flu = influenza_df.copy()
        df_flu['date'] = pd.to_datetime(df_flu['date'])
        
        # Resample influenza to weekly (week ending Sunday)
        df_flu = df_flu.set_index('date')
        df_flu_weekly = df_flu.resample('W')['influenza'].mean().reset_index()
        df_flu_weekly.columns = ['week', 'influenza']
        
        # Fill missing weeks
        min_week_flu = df_flu_weekly['week'].min()
        max_week_flu = df_flu_weekly['week'].max()
        complete_weeks_flu = pd.date_range(start=min_week_flu, end=max_week_flu, freq='W')
        complete_df_flu = pd.DataFrame({'week': complete_weeks_flu})
        df_flu_weekly = complete_df_flu.merge(df_flu_weekly, on='week', how='left')
        df_flu_weekly['influenza'] = df_flu_weekly['influenza'].ffill().bfill()
        
        print(f"\n🦠 Influenza Data:")
        print(f"  Weeks: {len(df_flu_weekly)}")
        print(f"  Date range: {df_flu_weekly['week'].min()} to {df_flu_weekly['week'].max()}")
        
        # 3. Merge appointments and influenza on week
        merged_df = df_apps.merge(df_flu_weekly, on='week', how='inner')
        print(f"\n🔗 Merged Data:")
        print(f"  Total weeks: {len(merged_df)}")
        print(f"  Date range: {merged_df['week'].min()} to {merged_df['week'].max()}")
        
        if len(merged_df) == 0:
            raise ValueError("No overlapping weeks between appointments and influenza data")
        
       # 4. Create Darts TimeSeries for appointments
        ts_appointments = TimeSeries.from_dataframe(
            merged_df[['week', 'appointments']], 
            time_col='week', 
            value_cols='appointments', 
            freq='W'
        )
        
        # 5. Create Darts TimeSeries for influenza
        ts_influenza = TimeSeries.from_dataframe(
            merged_df[['week', 'influenza']], 
            time_col='week', 
            value_cols='influenza', 
            freq='W'
        )
        
        # 6. Stack into multivariate series [appointments, influenza]
        train_ts = ts_appointments.stack(ts_influenza)
        
        # 7. Create metadata
        metadata = {
            'success': True,
            'total_weeks': len(train_ts),
            'start_date': train_ts.start_time(),
            'end_date': train_ts.end_time(),
            'training_weeks': len(merged_df),  # Total merged weeks
            'current_weeks': 0,  # Not applicable in this workflow
            'components': train_ts.components.tolist(),
            'message': f'Successfully merged {len(train_ts)} weeks of data',
            'appointments_series': ts_appointments  # Store for plotting
        }
        
        print(f"\n✅ Training Dataset Created:")
        print(f"  Total weeks: {len(train_ts)}")
        print(f"  Date range: {train_ts.start_time()} to {train_ts.end_time()}")
        print(f"  Components: {train_ts.components.tolist()}")
        print("="*60 + "\n")
        
        return train_ts, metadata
        
    except Exception as e:
        print(f"\n❌ Error merging training data: {str(e)}\n")
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
        
        # 8. Calculate validation metrics
        print(f"\n📊 Calculating Validation Metrics...")
        
        # Limit validation prediction to output_chunk_length (we don't have future covariates beyond that)
        val_pred_length = min(val_length, output_chunk_length)
        
        # Get validation data (last val_pred_length weeks)
        val_target = target_series[-val_pred_length:]
        
        # Generate predictions on validation set (limited to output_chunk_length)
        val_pred_scaled = model.predict(
            n=val_pred_length,
            series=target_series[:-val_pred_length],
            past_covariates=covariate_series[:-val_pred_length]
        )
        
        # Calculate metrics on scaled data
        val_smape = smape(val_target, val_pred_scaled)
        val_mape = mape(val_target, val_pred_scaled)
        val_rmse = rmse(val_target, val_pred_scaled)
        val_mae = mae(val_target, val_pred_scaled)
        
        print(f"  sMAPE: {val_smape:.2f}%")
        print(f"  MAPE: {val_mape:.2f}%")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        
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
        
        print(f"\n📦 Model Package Created:")
        print(f"  ✓ Trained NBEats model")
        print(f"  ✓ Target scaler")
        print(f"  ✓ Covariate scaler")
        print(f"  ✓ Scaled time series")
        print(f"  ✓ Validation metrics")
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
