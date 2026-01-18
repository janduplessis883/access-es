import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import streamlit as st
from notionhelper import NotionHelper
from darts.models import NBEATSModel
from darts.metrics import smape, mape, rmse, mae, r2_score
from darts.utils.likelihood_models import QuantileRegression
import matplotlib.dates as mdates
import holidays


# --- create_ts_data (kept logically the same) ---

def create_ts_data():
    # 1. Influenza Data from Notion
    nh = NotionHelper(st.secrets['NOTION_TOKEN'])
    data_id = st.secrets['DATA_ID']
    inf = nh.get_data_source_pages_as_dataframe(data_id)
    
    inf.drop(columns=['Name', 'notion_page_id'], inplace=True)
    inf.columns = ['inf', 'week']
    inf['week'] = pd.to_datetime(inf['week'], format='mixed', yearfirst=True)
    inf.sort_values(by='week', inplace=True)
    inf.dropna(inplace=True)
    
    # 2. Load and Combine Appointment Data
    filtered_df = pd.read_csv('/Users/janduplessis/Downloads/filtered_appointments_current.csv')
    h_data = pd.read_csv('/Users/janduplessis/Downloads/Historic_filtered_appointments.csv')
    
    filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date'], format='%Y-%m-%d')
    h_data['appointment_date'] = pd.to_datetime(h_data['appointment_date'], format='%Y-%m-%d')
    
    join_date = pd.Timestamp('2025-04-01')
    filtered_df_new = filtered_df[filtered_df['appointment_date'] >= join_date].copy()
    h_data_new = h_data[h_data['appointment_date'] < join_date].copy()
    
    full_train_apps = pd.concat([h_data_new, filtered_df_new], axis=0)
    full_train_apps.sort_values(by='appointment_date', inplace=True)
    
    # 3. Resample to Weekly Target
    full_train = full_train_apps.resample('W', on='appointment_date').size().reset_index(name='apps')
    full_train.columns = ['week', 'apps']
    
    # 4. Feature Engineering: 4-Week Rolling Mean
    full_train['apps_rolling'] = full_train['apps'].rolling(window=4, min_periods=1).mean()
    
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
        value_cols=['apps', 'inf', 'apps_rolling', 'work_days'],
        freq='W'
    )

    # Verify no NaN in the series
    print(f"Debug - Combined TS has NaN: {np.isnan(combined_ts.values()).any()}")
    
    return combined_ts


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
    cov_cols = ['inf', 'apps_rolling', 'work_days']
    
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
    
    print("\n📊 Model Performance Metrics:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df


def plot_forecast_result(full_series_unscaled, val_pred_orig, future_pred_orig, metric_score):
    """
    Plots the probabilistic forecast with holiday awareness.
    """
    print(f"Debug - Historical: {full_series_unscaled['apps'].start_time} to {full_series_unscaled['apps'].end_time}")
    print(f"Debug - Val pred: {val_pred_orig.start_time} to {val_pred_orig.end_time}")
    print(f"Debug - Future pred: {future_pred_orig.start_time} to {future_pred_orig.end_time}")

    fig, ax = plt.subplots(figsize=(16, 8))

    # 1. Plot Actual Apps (Gray) - full historical
    full_series_unscaled['apps'].plot(
        ax=ax, label='Actual Apps', color='#999999', alpha=0.4, linewidth=1
    )

    # 2. Plot 4-Week Rolling Average (Dashed Dark Gray)
    if 'apps_rolling' in full_series_unscaled.components:
        full_series_unscaled['apps_rolling'].plot(
            ax=ax, label='4-Week Rolling Avg',
            color='#444444', linestyle='--', linewidth=1.5, alpha=0.6
        )

    # 3. Plot Work Days (Capacity Indicator) on Secondary Axis
    ax2 = None
    if 'work_days' in full_series_unscaled.components:
        ax2 = ax.twinx()
        full_series_unscaled['work_days'].plot(
            ax=ax2, label='Work Days', color='red', linestyle=':', alpha=0.2
        )
        ax2.set_ylim(0, 6)
        ax2.set_ylabel("Working Days", color='red', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=8)

    # 4. Plot Backtest (Orange) - use Darts' built-in quantile plotting
    # First, ensure val_pred_orig is a deterministic series by taking the median
    val_median = val_pred_orig.quantile(0.5)
    val_median.plot(ax=ax, label='Backtest', color='#FF8C00', linewidth=2)

    # 5. Plot Future Forecast (Blue)
    future_median = future_pred_orig.quantile(0.5)
    future_median.plot(ax=ax, label='12-Week Forecast', color='#1f77b4', linewidth=2.5)

    # 6. Add uncertainty bands manually as shaded area
    val_p10 = val_pred_orig.quantile(0.1)
    val_p90 = val_pred_orig.quantile(0.9)
    ax.fill_between(
        list(val_p10.time_index), val_p10.values().flatten(), val_p90.values().flatten(),
        color='#FF8C00', alpha=0.15
    )

    future_p10 = future_pred_orig.quantile(0.1)
    future_p90 = future_pred_orig.quantile(0.9)
    ax.fill_between(
        list(future_p10.time_index), future_p10.values().flatten(), future_p90.values().flatten(),
        color='#1f77b4', alpha=0.15
    )

    # Add vertical line at forecast start
    if len(future_pred_orig) > 0:
        ax.axvline(
            x=future_median.start_time(),
            color='gray', linestyle='--', alpha=0.7, label='Forecast Start'
        )

    # Clean up the sMAPE display
    try:
        if pd.isna(metric_score):
            score_text = "Calculating..."
        else:
            score_text = f"{float(metric_score):.2f}%"
    except Exception:
        score_text = "N/A"

    ax.set_title(
        f"Probabilistic N-BEATS: Holiday Aware Model\nFinal sMAPE: {score_text}",
        fontsize=16, fontweight='bold'
    )

    ax.legend(loc='upper left')
    ax.grid(True, which='both', alpha=0.1)
    plt.tight_layout()
    plt.show()


# --- Model Training & Forecast ---

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
    
    cov_cols = ['inf', 'apps_rolling', 'work_days']
    past_cov_train = train_scaled[cov_cols]
    past_cov_full = series_scaled[cov_cols]

    # 2. Initialize Model with Quantile Likelihood
    model = NBEATSModel(
        input_chunk_length=user_params['input_chunk_length'],
        output_chunk_length=user_params['output_chunk_length'],
        generic_architecture=user_params['generic_architecture'],
        num_blocks=user_params['num_blocks'],
        num_layers=user_params['num_layers'],
        layer_widths=user_params['layer_widths'],  # can be int or list with len==num_stacks
        n_epochs=user_params['n_epochs'],
        batch_size=user_params['batch_size'],
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
        past_covariates=past_cov_train,
        num_samples=100
    )
    pred_future_scaled = model.predict(
        n=future_weeks,
        series=target_full,
        past_covariates=past_cov_full,
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
    plot_forecast_result(combined_ts, val_pred_orig, pred_future_orig, current_smape)
    
    return pred_future_orig, metrics_results, model


if __name__ == "__main__":
    print("🚧 Starting Script")
    
    # 1. Build time series with holiday-aware features
    combined = create_ts_data()

    # 2. Scale and Split
    train_scaled, val_scaled, series_scaled, target_scaler, cov_scaler = scale_and_split_ts(
        combined,
        test_size=12
    )
    print("💾 Time Series scaled")
    
    # 3. Define User Parameters
    selected_params = {
        "input_chunk_length": 52,
        "output_chunk_length": 12,
        "generic_architecture": False,
        "num_blocks": 3,
        "num_layers": 4,
        "layer_widths": 256,
        "n_epochs": 150,
        "batch_size": 64,
        "learning_rate": 0.0001  # Standard default is 1e-3
    }

    # 4. Run Forecast and Generate Plot
    forecast_series, metrics_results, trained_model = run_nbeats_forecast(
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        series_scaled=series_scaled,
        target_scaler=target_scaler,
        cov_scaler=cov_scaler,
        user_params=selected_params,
        combined_ts=combined,
        future_weeks=12
    )