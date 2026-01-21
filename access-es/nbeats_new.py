import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
import streamlit as st
from notionhelper import NotionHelper
from darts.models import NBEATSModel
from darts.metrics import smape, mape, rmse, mae, r2_score
from darts.utils.likelihood_models import QuantileRegression
import matplotlib.dates as mdates
import holidays
import plotly.graph_objects as go


# --- create_ts_data (kept logically the same) ---

def create_ts_data(filtered_df, h_data, inf_df=None):
    """
    Create time series data from appointment dataframes.

    Args:
        filtered_df: Current appointments DataFrame with 'appointment_date' column
        h_data: Historic appointments DataFrame with 'appointment_date' column
        inf_df: Optional influenza DataFrame (will fetch from Notion if None)

    Returns:
        tuple: (TimeSeries object, matplotlib.figure.Figure)
    """
    # 1. Influenza Data from Notion (if not provided)
    if inf_df is None:
        nh = NotionHelper(st.secrets['NOTION_TOKEN'])
        data_id = st.secrets['DATA_ID']
        inf = nh.get_data_source_pages_as_dataframe(data_id)

        inf.drop(columns=['Name', 'notion_page_id'], inplace=True)
        inf.columns = ['inf', 'week']
        inf['week'] = pd.to_datetime(inf['week'], format='mixed', yearfirst=True)
        inf.sort_values(by='week', inplace=True)
        inf.dropna(inplace=True)
    else:
        inf = inf_df.copy()

    # 2. Process Appointment Data
    filtered_df = filtered_df.copy()
    h_data = h_data.copy()
    try:
        filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date'], format='%Y-%m-%d')
        h_data['appointment_date'] = pd.to_datetime(h_data['appointment_date'], format='%Y-%m-%d')
    except Exception as e:
        filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date'], format='%d-%b-%y')
        h_data['appointment_date'] = pd.to_datetime(h_data['appointment_date'], format='%d-%b-%y')

    join_date = pd.Timestamp('2025-04-01')
    filtered_df_new = filtered_df[filtered_df['appointment_date'] >= join_date].copy()
    h_data_new = h_data[h_data['appointment_date'] < join_date].copy()

    full_train_apps = pd.concat([h_data_new, filtered_df_new], axis=0)
    full_train_apps.sort_values(by='appointment_date', inplace=True)

    # 3. Resample to Weekly Target
    full_train = full_train_apps.resample('W', on='appointment_date').size().reset_index(name='apps')
    full_train.columns = ['week', 'apps']

    # 4. Feature Engineering: 4-Week Rolling Mean
    # full_train['apps_rolling4'] = full_train['apps'].rolling(window=4, min_periods=1).mean()
    # full_train['apps_rolling8'] = full_train['apps'].rolling(window=8, min_periods=1).mean()
    full_train['apps_rolling4']  = full_train['apps'].ewm(alpha=0.4,  adjust=True, min_periods=1).mean()
    full_train['apps_rolling8']  = full_train['apps'].ewm(alpha=0.15,  adjust=True, min_periods=1).mean()

    # 5. Feature Engineering: Weekly Working Days (The "Holiday Fix")
    start_date = full_train['week'].min()
    end_date = full_train['week'].max() + pd.Timedelta(weeks=12)
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')

    df_days = pd.DataFrame({'date': all_days})
    df_days['is_weekday'] = df_days['date'].dt.dayofweek < 5

    # UK bank holidays – England
    uk_holidays = holidays.UnitedKingdom(subdiv='England')
    df_days['is_bank_holiday'] = df_days['date'].apply(lambda x: x in uk_holidays)

    # Working day if weekday and not bank holiday
    df_days['work_day_val'] = (df_days['is_weekday'] & ~df_days['is_bank_holiday']).astype(int)

    # Weekly work days
    weekly_work_days = df_days.resample('W', on='date')['work_day_val'].sum().reset_index()
    weekly_work_days.columns = ['week', 'work_days']

    # Merge into main data
    combined = inf.merge(full_train, on='week', how='inner')  # Use inner join to avoid NaNs
    combined = combined.merge(weekly_work_days, on='week', how='inner')

    # Drop any remaining NaN values
    combined = combined.dropna(subset=['apps', 'inf'])

    # Forward fill work_days if missing (shouldn't be but just in case)
    combined['work_days'] = combined['work_days'].ffill().bfill()
    combined['work_days'] = combined['work_days'].astype('float32')

    combined_ts = TimeSeries.from_dataframe(
        combined,
        time_col='week',
        value_cols=['apps', 'inf', 'apps_rolling4',  'apps_rolling8','work_days'],
        freq='W'
    )

    # Verify no NaN in the series
    print(f"Debug - Combined TS has NaN: {np.isnan(combined_ts.values()).any()}")

    # Create plot with distinct colors for each series
    fig, ax = plt.subplots(figsize=(16, 4))
    
    combined_ts['apps'].plot(ax=ax, label='Appointments', color='#c0c0c0', alpha=0.8, linewidth=1.2, zorder=1)
    # Plot other series as line plots
    combined_ts['inf'].plot(ax=ax, label='Influenza', color='#434346', alpha=0.7, linewidth=1.2, zorder=2)
    combined_ts['apps_rolling4'].plot(ax=ax, label='alpha 0.4 ewm', color='#b7516b', alpha=0.7, linewidth=1.2, zorder=2)
    combined_ts['apps_rolling8'].plot(ax=ax, label='alpha 0.15 ewm', color='#4a6977', alpha=0.7, linewidth=1.2, zorder=2)

    # Plot work_days on secondary y-axis (different scale)
    ax2 = ax.twinx()
    combined_ts['work_days'].plot(ax=ax2, label='Work Days', color='#ae4f4d', alpha=0.6, linewidth=1)
    ax2.set_ylabel('Work Days', color='#ae4f4d')
    ax2.tick_params(axis='y', labelcolor='#ae4f4d')
    ax2.set_ylim(0, 6)
    
    ax.set_title("Time Series Data with Holiday Awareness", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return combined_ts, fig


def scale_and_split_ts(combined_ts, test_size=12):
    """
    Splits and scales the multivariate TimeSeries.
    Now includes 'work_days' to account for holiday capacity drops.
    """
    # 1. Split into Train and Validation
    train, val = combined_ts[:-test_size], combined_ts[-test_size:]
    
    # 2. Initialize separate scalers for Target and Covariates
    target_scaler = Scaler()
    cov_scaler = Scaler()
    
    # 3. Fit and transform target (apps)
    target_scaler.fit(train['apps'])
    train_target = target_scaler.transform(train['apps'])
    val_target = target_scaler.transform(val['apps'])
    full_target = target_scaler.transform(combined_ts['apps'])
    
    # 4. Fit and transform covariates (inf, apps_rolling, AND work_days)
    cov_cols = ['inf', 'apps_rolling4', 'apps_rolling8', 'work_days']
    
    cov_scaler.fit(train[cov_cols])
    train_cov = cov_scaler.transform(train[cov_cols])
    val_cov = cov_scaler.transform(val[cov_cols])
    full_cov = cov_scaler.transform(combined_ts[cov_cols])
    
    # 5. Stack them back together into multivariate series
    train_scaled = train_target.stack(train_cov)
    val_scaled = val_target.stack(val_cov)
    series_scaled = full_target.stack(full_cov)
    
    # 6. Return all components for the training function
    return train_scaled, val_scaled, series_scaled, target_scaler, cov_scaler


# --- Metrics and Plotting ---

def calculate_metrics(val_original, val_pred_original):
    """
    Calculates various time series metrics comparing actuals vs predictions.
    """
    metrics_dict = {
        "Metric": ["sMAPE (%)", "MAPE (%)", "RMSE", "MAE", "R2 Score"],
        "Value": [
            smape(val_original, val_pred_original),
            mape(val_original, val_pred_original),
            rmse(val_original, val_pred_original),
            mae(val_original, val_pred_original),
            r2_score(val_original, val_pred_original)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    
    print("\nModel Performance Metrics:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df


def plot_forecast_result(full_series_unscaled, val_pred_orig, future_pred_orig, metric_score):
    """
    Plots the probabilistic forecast using Plotly for Streamlit.
    """
    fig = go.Figure()

    # 1. Actual Apps (Gray)
    actual_df = full_series_unscaled['apps'].to_dataframe()
    fig.add_trace(go.Scatter(
        x=actual_df.index, y=actual_df['apps'],
        name='Actual Apps', line=dict(color='#c0c0c0', width=2),
        mode='lines'
    ))

    # 2. Rolling Averages
    if 'apps_rolling4' in full_series_unscaled.components:
        r4 = full_series_unscaled['apps_rolling4'].to_dataframe()
        fig.add_trace(go.Scatter(
            x=r4.index, y=r4['apps_rolling4'],
            name='alpha 0.4 ewm', line=dict(color='#b7516b', width=1),
            opacity=0.7
        ))
    
    if 'apps_rolling8' in full_series_unscaled.components:
        r8 = full_series_unscaled['apps_rolling8'].to_dataframe()
        fig.add_trace(go.Scatter(
            x=r8.index, y=r8['apps_rolling8'],
            name='alpha 0.15 ewm', line=dict(color='#4a6977', width=1),
            opacity=0.7
        ))

    # 3. Work Days (Secondary Y-Axis)
    if 'work_days' in full_series_unscaled.components:
        wd = full_series_unscaled['work_days'].to_dataframe()
        fig.add_trace(go.Scatter(
            x=wd.index, y=wd['work_days'],
            name='Work Days', line=dict(color='#ae4f4d', width=0.5),
            yaxis='y2', opacity=0.5
        ))

    # 4. Backtest & Confidence Interval (Orange)
    val_median = val_pred_orig.quantile(0.5).to_series()
    val_p10 = val_pred_orig.quantile(0.1).to_series()
    val_p90 = val_pred_orig.quantile(0.9).to_series()

    fig.add_trace(go.Scatter(
        x=val_p10.index.tolist() + val_p90.index.tolist()[::-1],
        y=val_p10.tolist() + val_p90.tolist()[::-1],
        fill='toself', fillcolor='rgba(255, 140, 0, 0.15)',
        line=dict(color='rgba(255,140,0,0)'), name='Backtest CI', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=val_median.index, y=val_median,
        name='Backtest', line=dict(color='#FF8C00', width=1.5)
    ))

    # 5. 12-Week Forecast & Confidence Interval (Blue)
    fut_median = future_pred_orig.quantile(0.5).to_series()
    fut_p10 = future_pred_orig.quantile(0.1).to_series()
    fut_p90 = future_pred_orig.quantile(0.9).to_series()

    fig.add_trace(go.Scatter(
        x=fut_p10.index.tolist() + fut_p90.index.tolist()[::-1],
        y=fut_p10.tolist() + fut_p90.tolist()[::-1],
        fill='toself', fillcolor='rgba(31, 119, 180, 0.15)',
        line=dict(color='rgba(31,119,180,0)'), name='Forecast CI', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fut_median.index, y=fut_median,
        name='12-Week Forecast', line=dict(color='#1f77b4', width=2.5)
    ))

    # 6. Layout, Grid, and Titles
    score_text = f"{float(metric_score):.2f}%" if not pd.isna(metric_score) else "N/A"
    
    fig.update_layout(
        title=f"<b>Probabilistic N-BEATS: Holiday Aware Model</b><br>Final sMAPE: {score_text}",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=[pd.Timestamp('2025-01-01'), pd.Timestamp.now() + pd.DateOffset(weeks=12)],
            showgrid=True, gridwidth=0.3, gridcolor='rgba(0,0,0,0.1)', # Vertical Grid
            zeroline=False
        ),
        yaxis=dict(
            title="Apps",
            showgrid=True, gridwidth=0.3, gridcolor='rgba(0,0,0,0.1)', # Horizontal Grid
        ),
        yaxis2=dict(
            title="Working Days",
            overlaying='y', side='right', range=[0, 6],
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
        height=600
    )

    # Vertical Forecast Start Line
    if len(future_pred_orig) > 0:
        fig.add_vline(x=fut_median.index[0], line_width=1.5, line_dash="dash", line_color="gray")

    return fig


# --- Model Training & Forecast ---

def run_historical_forecast(
    model,
    series_scaled,
    val,
    target_scaler,
    forecast_horizon=7,
    stride=7
):
    """
    Generate historical forecasts (backtesting) using a sliding window approach.
    
    Args:
        model: Trained NBEATSModel
        series_scaled: Full scaled time series
        val: Validation portion of the time series
        target_scaler: Scaler for inverse transforming predictions
        forecast_horizon: Number of steps to forecast at each point (default: 7 days/1 week)
        stride: Number of steps to move forward between forecasts (default: 7 days/1 week)
    
    Returns:
        TimeSeries: Concatenated historical forecasts (inverse transformed)
    """
    from darts import concatenate
    
    print(f"⏳ Running historical forecasts with horizon={forecast_horizon}, stride={stride}")
    
    # Generate rolling forecasts
    pred_series = model.historical_forecasts(
        series_scaled,
        start=val.start_time(),
        forecast_horizon=forecast_horizon,
        stride=stride,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    
    # Concatenate all forecast segments
    pred_series_concat = concatenate(pred_series)
    
    # Inverse transform to original scale
    pred_series_original = target_scaler.inverse_transform(pred_series_concat)
    
    print(f"✅ Historical forecast complete. Generated {len(pred_series)} forecast segments")
    print(f"📊 Total forecast length: {len(pred_series_original)} time steps")
    
    return pred_series_original


def run_nbeats_forecast(
    train_scaled,
    val_scaled,
    series_scaled,
    target_scaler,
    cov_scaler,
    user_params,
    combined_ts,
    future_weeks=12
):
    """
    Trains N-BEATS with holiday/work_days awareness to improve sMAPE during 
    seasonal dips (Christmas/New Year).
    """
    print(f"⚙️ Training NBEATS with Holiday Awareness. Params: {user_params}")

    # 1. Slice Components
    target_train = train_scaled['apps']
    target_val = val_scaled['apps']
    target_full = series_scaled['apps']

    cov_cols = ['inf', 'apps_rolling4', 'apps_rolling8', 'work_days']
    past_cov_train = train_scaled[cov_cols]
    past_cov_full = series_scaled[cov_cols]

    # 2. Initialize Model with Quantile Likelihood
    model = NBEATSModel(
        input_chunk_length=user_params['input_chunk_length'],
        output_chunk_length=user_params['output_chunk_length'],
        generic_architecture=user_params['generic_architecture'],
        num_stacks=user_params.get('num_stacks', 30),
        num_blocks=user_params['num_blocks'],
        num_layers=user_params['num_layers'],
        layer_widths=user_params['layer_widths'],
        n_epochs=user_params['n_epochs'],
        batch_size=user_params['batch_size'],
        dropout=user_params.get('dropout', 0.1),
        optimizer_kwargs={'lr': user_params.get('learning_rate', 0.0005)},
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        nr_epochs_val_period=1,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu", "gradient_clip_val": 1.0}
    )

    # 3. Fit Model
    model.fit(
        series=target_train,
        past_covariates=past_cov_train,
        val_series=target_full,
        val_past_covariates=past_cov_full,
        verbose=True
    )

    # 4. Generate Probabilistic Predictions
    pred_val_scaled = model.predict(
        n=len(target_val),
        series=target_train,
        past_covariates=past_cov_full,  # Use full series to cover prediction period
        num_samples=100
    )
    pred_future_scaled = model.predict(
        n=future_weeks,
        series=target_full,
        past_covariates=past_cov_full,  # Already extended in create_ts_data
        num_samples=100
    )

    # 5. Inverse Transform
    print(f"Debug - pred_val_scaled has NaN: {np.isnan(pred_val_scaled.values()).any()}")
    print(f"Debug - pred_val_scaled sample values: {pred_val_scaled.values()[:3]}")
    val_pred_orig = target_scaler.inverse_transform(pred_val_scaled)
    val_actual_orig = target_scaler.inverse_transform(target_val)
    pred_future_orig = target_scaler.inverse_transform(pred_future_scaled)

    print(f"Debug - val_pred_orig has NaN: {np.isnan(val_pred_orig.values()).any()}")
    print(f"Debug - val_pred_orig values: {val_pred_orig.values()[:3]}")

    # 6. Calculate Metrics
    val_pred_orig = val_pred_orig.slice_intersect(val_actual_orig)
    metrics_results = calculate_metrics(val_actual_orig, val_pred_orig)

    # Pull sMAPE (first row in metrics table)
    current_smape = metrics_results["Value"][0]

    # 7. Plot Result using the *unscaled* original series
    print(f"Components available for plotting: {combined_ts.components}")
    print(f"Validation Prediction samples: {val_pred_orig.n_samples}")
    fig = plot_forecast_result(combined_ts, val_pred_orig, pred_future_orig, current_smape)

    return pred_future_orig, metrics_results, model, fig


def train_model_for_app(
    combined_ts,
    input_chunk_length=52,
    output_chunk_length=12,
    n_epochs=100,
    num_blocks=4,
    num_stacks=30,
    num_layers=4,
    layer_widths=512,
    batch_size=32,
    learning_rate=0.0005,
    dropout=0.1,
):
    """
    Train N-BEATS model for Streamlit app with user-specified hyperparameters.
    Validation is fixed at 24 weeks.
    
    Args:
        combined_ts: TimeSeries object with training data
        input_chunk_length: Length of input sequence
        output_chunk_length: Length of output sequence
        n_epochs: Number of training epochs
        num_blocks: Number of blocks in the model
        num_stacks: Number of stacks in the model
        num_layers: Number of layers per block
        layer_widths: Width of each layer
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        dropout: Dropout rate for regularization
    
    Returns:
        dict: Training results with model, scalers, metrics, and metadata
    """
    try:
        # Fixed validation size: 24 weeks
        total_weeks = len(combined_ts)
        test_size = 24
        
        # Scale and split data
        train_scaled, val_scaled, series_scaled, target_scaler, cov_scaler = scale_and_split_ts(
            combined_ts,
            test_size=test_size
        )
        
        # Define parameters
        user_params = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "generic_architecture": False,
            "num_stacks": num_stacks,
            "num_blocks": num_blocks,
            "num_layers": num_layers,
            "layer_widths": layer_widths,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout
        }
        
        # Train model
        forecast_series, metrics_results, trained_model, fig = run_nbeats_forecast(
            train_scaled=train_scaled,
            val_scaled=val_scaled,
            series_scaled=series_scaled,
            target_scaler=target_scaler,
            cov_scaler=cov_scaler,
            user_params=user_params,
            combined_ts=combined_ts,
            future_weeks=12
        )
        
        # Extract metrics
        metrics = {
            'smape': metrics_results["Value"][0],
            'mape': metrics_results["Value"][1],
            'rmse': metrics_results["Value"][2],
            'mae': metrics_results["Value"][3],
            'r2_score': metrics_results["Value"][4],
        }
        
        # Convert forecast_series to DataFrame for app use
        forecast_median = forecast_series.quantile(0.5)
        forecast_df = pd.DataFrame({
            'week': forecast_median.time_index,
            'forecasted_appointments': forecast_median.univariate_values()
        })
        
        return {
            'success': True,
            'model': trained_model,
            'target_scaler': target_scaler,
            'cov_scaler': cov_scaler,
            'train_scaled': train_scaled,
            'val_scaled': val_scaled,
            'series_scaled': series_scaled,
            'combined_ts': combined_ts,
            'metrics': metrics,
            'metrics_df': metrics_results,
            'training_weeks': len(train_scaled),
            'validation_weeks': len(val_scaled),
            'total_weeks': total_weeks,
            'figure': fig,
            'forecast_df': forecast_df,
            'forecast_series': forecast_series,
            'message': f'Model trained successfully with {len(train_scaled)} training weeks'
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Training failed: {str(e)}',
            'error': str(e)
        }

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
        "input_chunk_length": [26, 52, 64, 104],
        "output_chunk_length": [6, 12],
        "generic_architecture": [False, True],
        "num_blocks": [2, 3, 4, 5],
        "num_layers": [3, 4],
        "num_stacks": [30, 50],
        "layer_widths": [256, 512, 1024],
        "n_epochs": [100, 150],
        "nr_epochs_val_period": [1],
        "batch_size": [64, 800],
        "random_state": [42],
        "force_reset": [True],
        "save_checkpoints": [False],
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


def full_predict():
    print("🚧 Starting Script")

    # 1. Load data from files (for standalone testing)
    filtered_df = pd.read_csv('/Users/janduplessis/Downloads/filtered_appointments.csv')
    h_data = pd.read_csv('/Users/janduplessis/Library/CloudStorage/OneDrive-NHS/python-data/ACCESS BI/FILTERED_TRAIN_SET_startApril21.csv')

    # 2. Build time series with holiday-aware features
    combined_ts, _ = create_ts_data(filtered_df, h_data)

    # 2. Scale and Split
    train_scaled, val_scaled, series_scaled, target_scaler, cov_scaler = scale_and_split_ts(
        combined_ts,
        test_size=24
    )
    print("💾 Time Series scaled")

    # 3. Define User Parameters
    selected_params = {
        "input_chunk_length": 52,
        "output_chunk_length": 12,
        "generic_architecture": False,
        "num_blocks": 4,
        "num_layers": 4,
        "layer_widths": 512,
        "n_epochs": 150,
        "batch_size": 128,
        "learning_rate": 0.0001  # Standard default is 1e-3
    }

    # 4. Run Forecast and Generate Plot
    forecast_series, metrics_results, trained_model, fig = run_nbeats_forecast(
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        series_scaled=series_scaled,
        target_scaler=target_scaler,
        cov_scaler=cov_scaler,
        user_params=selected_params,
        combined_ts=combined_ts,
        future_weeks=12
    )
    plt.show()

if __name__ == "__main__":
    print("🚧 Starting Script")

    # 1. Load data from files (for standalone testing)
    filtered_df = pd.read_csv('/Users/janduplessis/Downloads/filtered_appointments.csv')
    h_data = pd.read_csv('/Users/janduplessis/Library/CloudStorage/OneDrive-NHS/python-data/ACCESS BI/FILTERED_TRAIN_SET_startApril21.csv')

    # 2. Build time series with holiday-aware features
    combined_ts, _ = create_ts_data(filtered_df, h_data)

    # 2. Scale and Split
    train_scaled, val_scaled, series_scaled, target_scaler, cov_scaler = scale_and_split_ts(
        combined_ts,
        test_size=24
    )
    
    best_model, best_params, best_score = find_best_nbeats(train_scaled, val_scaled, torch_kwargs={"pl_trainer_kwargs": {"accelerator": "cpu"}})
