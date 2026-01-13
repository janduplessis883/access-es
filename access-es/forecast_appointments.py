"""
N-BEATS Forecasting for Surgery Appointments
Adapted from nbeats_forecasting.py template
"""

import pandas as pd
import numpy as np
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
    from darts.dataprocessing.transformers import Scaler, BoxCox
    from darts.dataprocessing.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    print("Warning: Darts library not available. Forecasting disabled.")


def forecast_surgery_appointments(
    weekly_df,
    influenza_csv_path="data/influenza.csv",
    forecast_weeks=12,
    input_chunk_length=52,
    output_chunk_length=12,
    n_epochs=100,
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    batch_size=32,
    learning_rate=5e-4,
    random_state=58,
    validation_length=12,
    accelerator="cpu",
    training_data_path="data/traming_data.csv",
):
    """
    Forecast surgery appointments using N-BEATS with influenza as covariate.
    Automatically combines with historical training data if available.

    Parameters:
    -----------
    weekly_df : pd.DataFrame
        Weekly aggregated data with columns: 'week', 'total_appointments'
    influenza_csv_path : str
        Path to influenza data CSV
    forecast_weeks : int
        Number of weeks to forecast (default: 12 for ~3 months)
    training_data_path : str
        Path to historical training data (will be combined with weekly_df)

    Returns:
    --------
    pd.DataFrame
        DataFrame with forecasted appointments: columns ['week', 'forecasted_appointments']
        Returns None if forecasting fails or libraries not available
    """

    if not DARTS_AVAILABLE:
        print("Darts library not available. Skipping forecasting.")
        return None

    try:
        # 1. Prepare surgery appointments data - combine with historical if available
        all_data_dfs = []

        # Load historical training data if available
        if os.path.exists(training_data_path):
            hist_df = pd.read_csv(training_data_path)
            hist_df["Date"] = pd.to_datetime(hist_df["Date"], format="%d-%b-%y")
            hist_df = hist_df.rename(
                columns={"Date": "date", "used_apps": "appointments"}
            )
            # Aggregate to weekly
            hist_df["week"] = hist_df["date"].dt.to_period("W").dt.start_time
            hist_weekly = hist_df.groupby("week")["appointments"].sum().reset_index()
            hist_weekly.columns = ["date", "appointments"]
            all_data_dfs.append(hist_weekly)

        # Add current weekly data
        current_df = weekly_df[["week", "total_appointments"]].copy()
        current_df.columns = ["date", "appointments"]
        current_df["date"] = pd.to_datetime(current_df["date"])
        all_data_dfs.append(current_df)

        # Combine all data
        if len(all_data_dfs) > 0:
            appointments_df = pd.concat(all_data_dfs, ignore_index=True)
            appointments_df = appointments_df.sort_values("date").reset_index(drop=True)
            # Remove duplicates, keeping most recent
            appointments_df = appointments_df.drop_duplicates(
                subset=["date"], keep="last"
            )
        else:
            appointments_df = current_df

        # 2. Load influenza data
        if not os.path.exists(influenza_csv_path):
            print(
                f"Influenza data not found at {influenza_csv_path}. Using appointments only."
            )
            # Forecast without covariate
            return forecast_without_covariate(appointments_df, forecast_weeks)

        influenza_df = pd.read_csv(influenza_csv_path)
        influenza_df["date"] = pd.to_datetime(influenza_df["date"])
        influenza_df = influenza_df.sort_values("date")

        # 3. Merge appointments with influenza on date
        # Align dates - find common date range
        min_date = max(appointments_df["date"].min(), influenza_df["date"].min())
        max_date = min(appointments_df["date"].max(), influenza_df["date"].max())

        appointments_df = appointments_df[
            (appointments_df["date"] >= min_date)
            & (appointments_df["date"] <= max_date)
        ]
        influenza_df = influenza_df[
            (influenza_df["date"] >= min_date) & (influenza_df["date"] <= max_date)
        ]

        # Merge
        combined_df = pd.merge(appointments_df, influenza_df, on="date", how="inner")

        if len(combined_df) < input_chunk_length:
            print(
                f"Insufficient data for forecasting. Need at least {input_chunk_length} weeks."
            )
            return None

        # 4. Create Darts TimeSeries
        series = TimeSeries.from_dataframe(
            combined_df,
            time_col="date",
            value_cols=["appointments", "influenza"],
            freq="W",
        ).astype("float32")

        # Split target and covariate
        target_series = series.univariate_component(0)  # appointments
        covariate_series = series.univariate_component(1)  # influenza

        # 5. Apply transformations
        pipeline = Pipeline(
            [BoxCox(lmbda=0), Scaler(MinMaxScaler())]  # Log transformation
        )
        target_transformed = pipeline.fit_transform(target_series)

        covariate_pipeline = Pipeline([BoxCox(lmbda=0), Scaler(MinMaxScaler())])
        covariate_transformed = covariate_pipeline.fit_transform(covariate_series)

        # 6. Split train/validation
        train_length = len(target_transformed) - validation_length
        if train_length < input_chunk_length:
            print("Insufficient training data.")
            return None

        train_transformed = target_transformed[:train_length]
        train_covariate = covariate_transformed[:train_length]

        # 7. Initialize Early Stopping
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode="min",
        )

        # 8. Initialize and train model
        print(f"Training N-BEATS model to forecast {forecast_weeks} weeks...")
        model = NBEATSModel(
            input_chunk_length=min(input_chunk_length, train_length),
            output_chunk_length=min(output_chunk_length, forecast_weeks),
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={"lr": learning_rate},
            random_state=random_state,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "callbacks": [my_stopper],
                "enable_progress_bar": False,
            },
        )

        # Train on full transformed data
        model.fit(
            series=target_transformed,
            past_covariates=covariate_transformed,
            verbose=False,
        )

        # 9. Generate forecasts
        # Note: Using last known influenza values for future (naive approach)
        forecast_transformed = model.predict(
            n=forecast_weeks,
            series=target_transformed,
            past_covariates=covariate_transformed,
        )
        forecast = pipeline.inverse_transform(forecast_transformed)

        # 10. Prepare output dataframe
        forecast_df = forecast.pd_dataframe()
        forecast_df.reset_index(inplace=True)
        forecast_df.columns = ["week", "forecasted_appointments"]
        forecast_df["forecasted_appointments"] = forecast_df[
            "forecasted_appointments"
        ].round(0)

        print(f"Successfully forecasted {len(forecast_df)} weeks of appointments.")
        return forecast_df

    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        return None


def forecast_without_covariate(appointments_df, forecast_weeks):
    """Fallback: Simple exponential smoothing forecast without covariate"""
    try:
        series = TimeSeries.from_dataframe(
            appointments_df, time_col="date", value_cols=["appointments"], freq="W"
        ).astype("float32")

        # Simple pipeline
        pipeline = Pipeline([BoxCox(lmbda=0), Scaler(MinMaxScaler())])
        transformed = pipeline.fit_transform(series)

        # Simple model with reduced parameters
        model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=50,
            random_state=42,
            force_reset=True,
            pl_trainer_kwargs={"enable_progress_bar": False},
        )

        model.fit(series=transformed, verbose=False)
        forecast_transformed = model.predict(n=forecast_weeks)
        forecast = pipeline.inverse_transform(forecast_transformed)

        forecast_df = forecast.pd_dataframe()
        forecast_df.reset_index(inplace=True)
        forecast_df.columns = ["week", "forecasted_appointments"]
        forecast_df["forecasted_appointments"] = forecast_df[
            "forecasted_appointments"
        ].round(0)

        return forecast_df

    except Exception as e:
        print(f"Simple forecasting also failed: {str(e)}")
        return None
