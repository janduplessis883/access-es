"""
Model Training Functions for NBEats Forecasting
Combines historical training data with current appointments
"""

import pandas as pd
import numpy as np
import os
import warnings

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


def parse_dates_flexible(df, date_column):
    """
    Parse dates with multiple format attempts.
    Returns parsed dates series and format used.
    """
    date_formats = [
        "%d-%b-%y",  # 01-Jan-22 (training data format)
        "%d-%m-%Y",  # 01-01-1985
        "%m-%d-%Y",  # 01-01-1985 (alternative)
        "%Y-%m-%d",  # 1985-01-01
        "%m/%d/%Y %H:%M",  # 2/24/2003 0:00
        "%d/%m/%Y %H:%M",  # 24/2/2003 0:00
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",  # 01.01.1985
        "%Y.%m.%d",  # 1985.01.01
    ]

    parsed_successfully = False
    successful_format = None
    parsed_dates = None

    for fmt in date_formats:
        try:
            test_parse = pd.to_datetime(df[date_column], format=fmt, errors="coerce")
            # Check if at least 50% of dates were parsed
            valid_count = test_parse.notna().sum()
            if valid_count > len(df) * 0.5:
                parsed_dates = test_parse
                parsed_successfully = True
                successful_format = fmt
                print(
                    f"Successfully parsed {valid_count}/{len(df)} dates using format: {fmt}"
                )
                break
        except:
            continue

    # If no format worked, try automatic parsing
    if not parsed_successfully:
        try:
            parsed_dates = pd.to_datetime(
                df[date_column], errors="coerce", infer_datetime_format=True
            )
            valid_count = parsed_dates.notna().sum()
            if valid_count > len(df) * 0.5:
                parsed_successfully = True
                successful_format = "auto-detected"
                print(
                    f"Successfully parsed {valid_count}/{len(df)} dates using auto-detection"
                )
        except:
            pass

    if not parsed_successfully or parsed_dates is None:
        sample_values = df[date_column].dropna().head(5).tolist()
        raise ValueError(
            f"Could not parse dates in column '{date_column}'. Sample values: {sample_values}"
        )

    return parsed_dates, successful_format


def prepare_training_data_robust(
    training_files, date_column="Date", appointments_column="used_apps"
):
    """
    Load and prepare user-uploaded training data with robust date parsing.
    Aggregates to weekly level.

    Parameters:
    -----------
    training_files : list
        List of uploaded training file objects (from st.file_uploader)
    date_column : str
        Name of the date column
    appointments_column : str
        Name of the appointments column

    Returns:
    --------
    pd.DataFrame
        Weekly aggregated data with columns ['week', 'total_appointments']
    """

    if not training_files or len(training_files) == 0:
        print("No training files provided")
        return None

    all_dfs = []

    for file in training_files:
        try:
            # Load training data from uploaded file
            print(f"Loading training data from {file.name}...")
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file)

            # Check columns exist
            if date_column not in df.columns or appointments_column not in df.columns:
                print(
                    f"Required columns not found in {file.name}. Available: {df.columns.tolist()}"
                )
                continue

            # Parse dates with flexible format handling
            df["date"], date_format = parse_dates_flexible(df, date_column)

            # Keep only rows with valid dates and appointments
            initial_count = len(df)
            df = df.dropna(subset=["date", appointments_column])
            removed_count = initial_count - len(df)

            if removed_count > 0:
                print(
                    f"Removed {removed_count} rows with invalid dates or missing appointments from {file.name}"
                )

            if len(df) == 0:
                print(f"No valid data in {file.name} after cleaning")
                continue

            # Rename appointments column for consistency
            df = df.rename(columns={appointments_column: "appointments"})
            all_dfs.append(df[["date", "appointments"]])

        except Exception as e:
            print(f"Error loading {file.name}: {str(e)}")
            continue

    if not all_dfs:
        print("No valid training data could be loaded")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values("date")

    # Aggregate to weekly level
    combined_df["week"] = combined_df["date"].dt.to_period("W").dt.start_time
    weekly_df = combined_df.groupby("week")["appointments"].sum().reset_index()
    weekly_df.columns = ["week", "total_appointments"]

    print(
        f"Successfully prepared {len(weekly_df)} weeks of training data from {len(all_dfs)} files"
    )
    print(f"Date range: {weekly_df['week'].min()} to {weekly_df['week'].max()}")

    return weekly_df


def prepare_influenza_data(influenza_csv_path="data/influenza.csv"):
    """
    Load and prepare influenza covariate data.
    Aggregates to weekly level.

    Parameters:
    -----------
    influenza_csv_path : str
        Path to influenza data CSV

    Returns:
    --------
    pd.DataFrame
        Weekly aggregated influenza data with columns ['week', 'influenza']
    """

    if not os.path.exists(influenza_csv_path):
        print(f"Influenza data not found at {influenza_csv_path}")
        return None

    try:
        # Load influenza data
        print(f"Loading influenza data from {influenza_csv_path}...")
        df = pd.read_csv(influenza_csv_path)

        # Parse dates with flexible format handling
        df["date"], date_format = parse_dates_flexible(df, "date")

        # Remove invalid dates
        df = df.dropna(subset=["date", "influenza"])

        if len(df) == 0:
            print("No valid influenza data after cleaning")
            return None

        # Aggregate to weekly level (take mean if multiple values per week)
        df["week"] = df["date"].dt.to_period("W").dt.start_time
        weekly_influenza = df.groupby("week")["influenza"].mean().reset_index()

        print(f"Successfully prepared {len(weekly_influenza)} weeks of influenza data")
        print(
            f"Date range: {weekly_influenza['week'].min()} to {weekly_influenza['week'].max()}"
        )

        return weekly_influenza

    except Exception as e:
        print(f"Error preparing influenza data: {str(e)}")
        return None


# TODO: Add function to combine training data with current appointment data
# TODO: Add function to merge with influenza covariate
# TODO: Add function to prepare final training dataset


def prepare_training_data(
    filtered_df,
    training_data_path="data/traming_data.csv",
    additional_training_files=None,
    date_column="Date",
    appointments_column="used_apps",
):
    """
    Combine historical training data with current filtered appointments.

    Parameters:
    -----------
    filtered_df : pd.DataFrame
        Current appointment data with appointment_date column
    training_data_path : str
        Path to historical training data CSV
    additional_training_files : list
        Optional list of additional CSV files to include
    date_column : str
        Name of the date column in training data CSVs
    appointments_column : str
        Name of the appointments column in training data CSVs

    Returns:
    --------
    pd.DataFrame
        Combined weekly aggregated data with columns ['week', 'total_appointments']
    """

    # 1. Load historical training data
    all_training_dfs = []

    if os.path.exists(training_data_path):
        hist_df = pd.read_csv(training_data_path)
        # Use user-specified column names
        if date_column in hist_df.columns and appointments_column in hist_df.columns:
            hist_df[date_column] = pd.to_datetime(
                hist_df[date_column], format="%d-%b-%y", errors="coerce"
            )
            hist_df = hist_df.rename(
                columns={date_column: "date", appointments_column: "appointments"}
            )
            # Filter out any rows with NaT dates
            hist_df = hist_df.dropna(subset=["date"])
            all_training_dfs.append(hist_df[["date", "appointments"]])
        else:
            print(
                f"Warning: Columns '{date_column}' or '{appointments_column}' not found in {training_data_path}"
            )

    # 2. Process current filtered data
    if filtered_df is not None and len(filtered_df) > 0:
        current_df = filtered_df[["appointment_date"]].copy()
        current_df["date"] = pd.to_datetime(current_df["appointment_date"])
        # Aggregate to daily level
        daily_current = (
            current_df.groupby("date").size().reset_index(name="appointments")
        )
        all_training_dfs.append(daily_current)

    # 3. Load additional training files if provided
    if additional_training_files:
        for file in additional_training_files:
            try:
                add_df = pd.read_csv(file)
                # Try to identify date and appointment columns
                date_col = [c for c in add_df.columns if "date" in c.lower()][0]
                app_col = [
                    c
                    for c in add_df.columns
                    if "app" in c.lower() or "used" in c.lower()
                ][0]

                add_df[date_col] = pd.to_datetime(add_df[date_col])
                add_df = add_df.rename(
                    columns={date_col: "date", app_col: "appointments"}
                )
                all_training_dfs.append(add_df[["date", "appointments"]])
            except Exception as e:
                print(f"Error loading additional training file: {e}")
                continue

    # 4. Combine all training data
    if not all_training_dfs:
        return None

    combined_daily = pd.concat(all_training_dfs, ignore_index=True)
    combined_daily = combined_daily.sort_values("date").reset_index(drop=True)

    # 5. Aggregate to weekly level
    combined_daily["week"] = combined_daily["date"].dt.to_period("W").dt.start_time
    weekly_training = combined_daily.groupby("week")["appointments"].sum().reset_index()
    weekly_training.columns = ["week", "total_appointments"]

    return weekly_training


def train_nbeats_model(
    weekly_training_df,
    influenza_csv_path="data/influenza.csv",
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
):
    """
    Train NBEats model on prepared training data.

    Parameters:
    -----------
    weekly_training_df : pd.DataFrame
        Weekly aggregated training data
    influenza_csv_path : str
        Path to influenza covariate data

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': trained NBEATSModel
        - 'pipeline': transformation pipeline
        - 'covariate_pipeline': covariate transformation pipeline
        -' training_series': transformed training series
        - 'covariate_series': transformed covariate series
        - 'success': bool
        - 'message': str
        - 'training_weeks': int
    """

    if not DARTS_AVAILABLE:
        return {
            "success": False,
            "message": "Darts library not available",
            "model": None,
        }

    try:
        # Check minimum data requirements
        if len(weekly_training_df) < input_chunk_length:
            return {
                "success": False,
                "message": f"Insufficient training data. Need at least {input_chunk_length} weeks, got {len(weekly_training_df)}",
                "model": None,
                "training_weeks": len(weekly_training_df),
            }

        # Load influenza data
        if not os.path.exists(influenza_csv_path):
            return {
                "success": False,
                "message": f"Influenza data not found at {influenza_csv_path}",
                "model": None,
            }

        influenza_df = pd.read_csv(influenza_csv_path)
        influenza_df["date"] = pd.to_datetime(influenza_df["date"])
        influenza_df["week"] = influenza_df["date"].dt.to_period("W").dt.start_time

        # Aggregate influenza to weekly (take mean if multiple values per week)
        influenza_weekly = (
            influenza_df.groupby("week")["influenza"].mean().reset_index()
        )

        # Merge training data with influenza
        merged_df = pd.merge(
            weekly_training_df, influenza_weekly, on="week", how="inner"
        )

        if len(merged_df) < input_chunk_length:
            return {
                "success": False,
                "message": f"Insufficient overlapping data after merging with influenza. Need {input_chunk_length} weeks, got {len(merged_df)}",
                "model": None,
                "training_weeks": len(merged_df),
            }

        # Create Darts TimeSeries
        series = TimeSeries.from_dataframe(
            merged_df,
            time_col="week",
            value_cols=["total_appointments", "influenza"],
            freq="W",
        ).astype("float32")

        # Split target and covariate
        target_series = series.univariate_component(0)  # appointments
        covariate_series = series.univariate_component(1)  # influenza

        # Apply transformations
        pipeline = Pipeline(
            [BoxCox(lmbda=0), Scaler(MinMaxScaler())]  # Log transformation
        )
        target_transformed = pipeline.fit_transform(target_series)

        covariate_pipeline = Pipeline([BoxCox(lmbda=0), Scaler(MinMaxScaler())])
        covariate_transformed = covariate_pipeline.fit_transform(covariate_series)

        # Initialize Early Stopping
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=15,
            min_delta=0.0001,
            mode="min",
        )

        # Initialize and train model
        model = NBEATSModel(
            input_chunk_length=min(
                input_chunk_length, len(target_transformed) - validation_length
            ),
            output_chunk_length=output_chunk_length,
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

        # Train the model
        model.fit(
            series=target_transformed,
            past_covariates=covariate_transformed,
            verbose=False,
        )

        return {
            "success": True,
            "message": f"Model trained successfully on {len(merged_df)} weeks of data",
            "model": model,
            "pipeline": pipeline,
            "covariate_pipeline": covariate_pipeline,
            "training_series": target_transformed,
            "covariate_series": covariate_transformed,
            "training_weeks": len(merged_df),
            "merged_data": merged_df,
        }

    except Exception as e:
        return {"success": False, "message": f"Training error: {str(e)}", "model": None}


def forecast_with_trained_model(trained_model_dict, forecast_weeks=12):
    """
    Generate forecasts using a trained model.

    Parameters:
    -----------
    trained_model_dict : dict
        Dictionary returned from train_nbeats_model()
    forecast_weeks : int
        Number of weeks to forecast

    Returns:
    --------
    pd.DataFrame
        Forecast dataframe with columns ['week', 'forecasted_appointments']
    """

    if not trained_model_dict or not trained_model_dict.get("success"):
        return None

    try:
        model = trained_model_dict["model"]
        pipeline = trained_model_dict["pipeline"]
        covariate_pipeline = trained_model_dict["covariate_pipeline"]
        training_series = trained_model_dict["training_series"]
        covariate_series = trained_model_dict["covariate_series"]

        # Generate forecast
        forecast_transformed = model.predict(
            n=forecast_weeks, series=training_series, past_covariates=covariate_series
        )

        # Inverse transform
        forecast = pipeline.inverse_transform(forecast_transformed)

        # Prepare output dataframe
        forecast_df = forecast.pd_dataframe()
        forecast_df.reset_index(inplace=True)
        forecast_df.columns = ["week", "forecasted_appointments"]
        forecast_df["forecasted_appointments"] = (
            forecast_df["forecasted_appointments"].round(0).clip(lower=0)
        )

        return forecast_df

    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        return None
