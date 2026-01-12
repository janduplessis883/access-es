"""
Utility functions for Access ES Appointment Tracker
Contains all data processing, calculations, and aggregations
"""

import pandas as pd
import numpy as np
from config import *


def load_and_combine_csv_files(uploaded_files):
    """Load and concatenate multiple CSV files"""
    dataframes = []
    file_info = []
    
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            dataframes.append(df)
            file_info.append({
                'name': uploaded_file.name,
                'rows': len(df),
                'success': True
            })
        except Exception as e:
            file_info.append({
                'name': uploaded_file.name,
                'error': str(e),
                'success': False
            })
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df, file_info
    
    return None, file_info


def extract_duration_minutes(duration_series):
    """Extract duration from format '1h 30m' to total minutes"""
    extracted = duration_series.str.extract(r'(?:(?P<hours>\d+)h)?\s?(?:(?P<minutes>\d+)m)?')
    hours = pd.to_numeric(extracted['hours']).fillna(0)
    minutes = pd.to_numeric(extracted['minutes']).fillna(0)
    return (hours * 60) + minutes


def preprocess_dataframe(df):
    """Clean and preprocess the combined dataframe"""
    # Process column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    
    # Rename columns
    df.rename(columns=COLUMN_RENAMES, inplace=True)
    
    # Convert appointment_date to datetime
    if 'appointment_date' in df.columns:
        try:
            df['appointment_date'] = pd.to_datetime(
                df['appointment_date'],
                format='%d-%b-%y'
            )
        except ValueError:
            df['appointment_date'] = pd.to_datetime(
                df['appointment_date'],
                format='mixed',
                dayfirst=True
            )
        
        # Drop invalid dates
        original_rows = len(df)
        df = df.dropna(subset=['appointment_date'])
        rows_dropped = original_rows - len(df)
        
        # Sort by date
        df = df.sort_values('appointment_date').reset_index(drop=True)
        
        # Convert duration columns
        if 'duration' in df.columns:
            df['duration'] = extract_duration_minutes(df['duration'])
        else:
            df['duration'] = 0
        
        if 'book_to_app' in df.columns:
            df['book_to_app'] = extract_duration_minutes(df['book_to_app'])
        else:
            df['book_to_app'] = 0
        
        return df, rows_dropped
    
    return df, 0


def filter_dataframe(df, selected_clinicians, selected_rota_types, date_range=None, exclude_dna=True):
    """Apply filters to dataframe"""
    # Filter by clinicians and rota types
    filtered_df = df[
        (df['clinician'].isin(selected_clinicians)) &
        (df['rota_type'].isin(selected_rota_types))
    ]
    
    # Filter by date range (if provided)
    if date_range is not None:
        filtered_df = filtered_df[
            (filtered_df['appointment_date'].dt.date >= date_range[0]) &
            (filtered_df['appointment_date'].dt.date <= date_range[1])
        ]
    
    # Store copy before DNA exclusion for stats
    filtered_df_with_dna = filtered_df.copy()
    
    # Exclude DNAs if requested
    dna_count = 0
    dna_percentage = 0
    if exclude_dna:
        dna_count_before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['appointment_status'] != 'Did Not Attend']
        dna_count = dna_count_before - len(filtered_df)
        dna_percentage = (dna_count / dna_count_before * 100) if dna_count_before > 0 else 0
    
    return filtered_df, filtered_df_with_dna, dna_count, dna_percentage


def calculate_time_metrics(date_range):
    """Calculate weeks and months from date range"""
    d_start = pd.Timestamp(date_range[0])
    d_end = pd.Timestamp(date_range[1])
    time_diff = (d_end - d_start).days
    weeks = time_diff / DAYS_PER_WEEK
    months = time_diff / DAYS_PER_MONTH
    
    return {
        'time_diff_days': time_diff,
        'weeks': weeks,
        'months': months,
        'safe_weeks': max(SAFE_WEEKS_MIN, weeks)
    }


def calculate_arrs_values(arrs_2526, arrs_month, arrs_future, slider_end_date, 
                         arrs_end_date, weeks_elapsed):
    """Calculate all ARRS-related values"""
    if arrs_2526 == 0:
        return {
            'estimated_weekly_arrs': 0.0,
            'future_arrs_apps': 0,
            'should_apply_arrs': False
        }
    
    # Calculate estimated weekly ARRS
    arrs_start_date = FY_START
    arrs_weeks_span = max(0.1, (arrs_end_date - arrs_start_date).days / DAYS_PER_WEEK)
    estimated_weekly_arrs = arrs_2526 / arrs_weeks_span
    
    # Determine if ARRS should be applied
    slider_end_ts = pd.Timestamp(slider_end_date) if not isinstance(slider_end_date, pd.Timestamp) else slider_end_date
    should_apply_arrs = slider_end_ts >= arrs_end_date
    
    # Calculate future ARRS if enabled
    future_arrs_apps = 0
    if arrs_future and should_apply_arrs:
        days_fut = (slider_end_ts - arrs_end_date).days
        future_weeks = days_fut / DAYS_PER_WEEK
        future_arrs_apps = int(round(estimated_weekly_arrs * future_weeks, 0)) if future_weeks > 0 else 0
    
    return {
        'estimated_weekly_arrs': estimated_weekly_arrs,
        'future_arrs_apps': future_arrs_apps,
        'should_apply_arrs': should_apply_arrs
    }


def calculate_apps_per_1000(total_apps, list_size, weeks):
    """Calculate appointments per 1000 patients per week"""
    safe_weeks = max(SAFE_WEEKS_MIN, weeks)
    return (total_apps / list_size) * 1000 / safe_weeks


def create_weekly_aggregation(df, list_size, arrs_params, cutoff_date):
    """Create weekly aggregation with ARRS and forecasted_apps column"""
    weekly_df = df.copy()
    weekly_df['week'] = weekly_df['appointment_date'].dt.to_period('W').dt.start_time
    weekly_agg = weekly_df.groupby('week').size().reset_index(name='total_appointments')
    
    weekly_agg['per_1000'] = weekly_agg['total_appointments'] / list_size * 1000
    
    # Calculate ARRS distribution
    weeks_after_cutoff = weekly_agg[weekly_agg['week'] >= cutoff_date].shape[0]
    
    if arrs_params['arrs_future'] and weeks_after_cutoff > 0:
        post_cutoff_increment = arrs_params['future_arrs_apps'] / weeks_after_cutoff
    else:
        post_cutoff_increment = 0
    
    # Split ARRS into historical and future
    weekly_agg['arrs_historical'] = np.where(
        weekly_agg['week'] < cutoff_date,
        arrs_params['estimated_weekly_arrs'],
        0
    )
    weekly_agg['arrs_future'] = np.where(
        weekly_agg['week'] >= cutoff_date,
        post_cutoff_increment,
        0
    )
    
    # Add forecasted_apps column (placeholder for future ML predictions)
    weekly_agg['forecasted_apps'] = 0
    
    # Calculate totals
    weekly_agg['arrs_only'] = weekly_agg['arrs_historical'] + weekly_agg['arrs_future']
    weekly_agg['total_with_arrs'] = weekly_agg['total_appointments'] + weekly_agg['arrs_only']
    weekly_agg['per_1000_with_arrs'] = weekly_agg['per_1000'] + (weekly_agg['arrs_only'] / list_size * 1000)
    
    # Total including forecasts (when implemented)
    weekly_agg['total_with_forecast'] = weekly_agg['total_with_arrs'] + weekly_agg['forecasted_apps']
    
    return weekly_agg


def create_monthly_aggregation(df, list_size, arrs_params, cutoff_date):
    """Create monthly aggregation with ARRS and forecasted_apps column"""
    monthly_df = df.copy()
    monthly_df['month'] = monthly_df['appointment_date'].dt.to_period('M').dt.start_time
    monthly_agg = monthly_df.groupby('month').size().reset_index(name='total_appointments')
    
    # Calculate months after cutoff
    months_after_cutoff = monthly_agg[monthly_agg['month'] >= cutoff_date].shape[0]
    
    if arrs_params['arrs_future'] and months_after_cutoff > 0:
        monthly_post_cutoff_increment = arrs_params['future_arrs_apps'] / months_after_cutoff
    else:
        monthly_post_cutoff_increment = 0
    
    # Split ARRS into historical and future
    monthly_agg['arrs_historical'] = np.where(
        monthly_agg['month'] < cutoff_date,
        (arrs_params['estimated_weekly_arrs'] * WEEKS_PER_MONTH),
        0
    )
    monthly_agg['arrs_future'] = np.where(
        monthly_agg['month'] >= cutoff_date,
        monthly_post_cutoff_increment,
        0
    )
    
    # Add forecasted_apps column (placeholder for future ML predictions)
    monthly_agg['forecasted_apps'] = 0
    
    # Calculate totals
    monthly_agg['total_arrs_estimated'] = monthly_agg['arrs_historical'] + monthly_agg['arrs_future']
    monthly_agg['total_with_arrs'] = monthly_agg['total_appointments'] + monthly_agg['total_arrs_estimated']
    
    # Total including forecasts (when implemented)
    monthly_agg['total_with_forecast'] = monthly_agg['total_with_arrs'] + monthly_agg['forecasted_apps']
    
    return monthly_agg


def calculate_clinician_stats(df_with_dna, selected_clinicians):
    """Calculate statistics per clinician"""
    clinician_stats = []
    
    for clinician in selected_clinicians:
        c_df = df_with_dna[df_with_dna['clinician'] == clinician]
        if not c_df.empty:
            total_apps = len(c_df)
            c_apps_df = c_df[c_df['appointment_status'] == 'Did Not Attend']
            dna_count = len(c_apps_df)
            dna_percentage = (dna_count / total_apps) * 100 if total_apps > 0 else 0
            duration = c_df['duration'].mean()
            book_to_app = c_df['book_to_app'].mean()
            
            clinician_stats.append({
                'Clinician': clinician,
                'Total Apps': total_apps,
                'DNAs': dna_count,
                'DNA %': round(dna_percentage, 2),
                'Avg App Duration (mins)': round(duration, 2),
                'Avg Book to App (mins)': round(book_to_app, 2)
            })
    
    if clinician_stats:
        stats_df = pd.DataFrame(clinician_stats)
        stats_df = stats_df.sort_values(by='Total Apps', ascending=False).reset_index(drop=True)
        return stats_df
    
    return None


def calculate_target_achievement(df, list_size, arrs_values, exp_add_apps_per_week=0):
    """Calculate target achievement metrics for FY"""
    last_data_date = pd.Timestamp(df['appointment_date'].max())
    data_start_date = pd.Timestamp(df['appointment_date'].min())
    
    if last_data_date >= FY_END:
        return None
    
    # Calculate time periods
    calc_start_date = min(FY_START, data_start_date)
    days_elapsed = (last_data_date - calc_start_date).days
    weeks_elapsed = max(0.1, days_elapsed / DAYS_PER_WEEK)
    
    days_left = (FY_END - last_data_date).days
    weeks_remaining = max(0.1, days_left / DAYS_PER_WEEK)
    
    # Calculate projections
    avg_weekly_surgery = len(df) / weeks_elapsed
    projected_surgery_baseline = avg_weekly_surgery * weeks_remaining
    
    # ARRS projection
    arrs_end_date = arrs_values.get('arrs_end_date', FY_END)
    weeks_from_arrs_to_end = max(0, (FY_END - arrs_end_date).days / DAYS_PER_WEEK)
    projected_future_arrs = arrs_values['estimated_weekly_arrs'] * weeks_from_arrs_to_end
    
    # Annual target
    annual_target_total = THRESHOLD_100_PERCENT * (list_size / 1000) * WEEKS_PER_YEAR
    
    # Total baseline projection
    achieved_so_far = len(df) + arrs_values.get('arrs_2526', 0)
    total_baseline_projection = achieved_so_far + projected_surgery_baseline + projected_future_arrs
    
    # Gap calculation
    gap = annual_target_total - total_baseline_projection
    required_extra_per_week = gap / weeks_remaining if weeks_remaining > 0 else 0
    
    return {
        'annual_target_total': annual_target_total,
        'total_baseline_projection': total_baseline_projection,
        'gap': gap,
        'required_extra_per_week': required_extra_per_week,
        'weeks_remaining': weeks_remaining,
        'weeks_elapsed': weeks_elapsed,
        'avg_weekly_surgery': avg_weekly_surgery,
        'projected_future_arrs': projected_future_arrs,
        'achieved_so_far': achieved_so_far,
        'exp_add_apps_per_week': exp_add_apps_per_week
    }


def create_projection_dataframe(weekly_agg, target_metrics, arrs_values, exp_add_apps_per_week, forecast_df=None):
    """Create dataframe for projection chart with optional ML forecasts"""
    # Historical part
    proj_df = weekly_agg[['week', 'total_appointments']].copy()
    proj_df['type'] = 'Historical'
    proj_df['ARRS'] = arrs_values['estimated_weekly_arrs']
    proj_df['Added (Exp)'] = 0
    proj_df['Catch-up Needed'] = 0
    
    # Future part
    last_historical_week = weekly_agg['week'].max()
    future_weeks_list = pd.date_range(
        start=last_historical_week + pd.Timedelta(weeks=1),
        end=FY_END,
        freq='W-MON'
    )
    
    # Create a mapping of week to forecast if available
    forecast_map = {}
    forecast_available = False
    
    if forecast_df is not None and 'week' in forecast_df.columns and len(forecast_df) > 0:
        forecast_available = True
        # Convert forecast weeks to datetime if needed
        forecast_df_copy = forecast_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(forecast_df_copy['week']):
            forecast_df_copy['week'] = pd.to_datetime(forecast_df_copy['week'])
        
        # Create mapping with better key handling
        for _, row in forecast_df_copy.iterrows():
            # Use week start (Monday) as key for consistency
            week_key = pd.Timestamp(row['week']).to_period('W').start_time
            forecast_map[week_key] = float(row['forecasted_appointments'])
        
        print(f"📊 Forecast mapping created: {len(forecast_map)} weeks")
        print(f"   First forecast week: {min(forecast_map.keys())}")
        print(f"   Last forecast week: {max(forecast_map.keys())}")
    
    future_data = []
    matched_count = 0
    
    for w in future_weeks_list:
        remaining_gap = max(0, target_metrics['required_extra_per_week'] - exp_add_apps_per_week)
        
        # Normalize to week start (Monday) for comparison
        week_key = pd.Timestamp(w).to_period('W').start_time
        
        # Use forecast if available, otherwise use average
        if week_key in forecast_map:
            forecasted_value = forecast_map[week_key]
            forecast_type = 'Forecasted'
            matched_count += 1
        else:
            forecasted_value = target_metrics['avg_weekly_surgery']
            forecast_type = 'Projected Baseline' if not forecast_available else 'Forecasted'
        
        future_data.append({
            'week': w,
            'total_appointments': forecasted_value,
            'type': forecast_type,
            'ARRS': arrs_values['estimated_weekly_arrs'],
            'Added (Exp)': exp_add_apps_per_week,
            'Catch-up Needed': remaining_gap
        })
    
    if future_data:
        future_df = pd.DataFrame(future_data)
        combined_proj_df = pd.concat([proj_df, future_df], ignore_index=True)
        
        # Print matching summary
        if forecast_available:
            print(f"✅ Matched {matched_count} out of {len(future_weeks_list)} projection weeks with forecast data")
        
        return combined_proj_df
    
    return proj_df
