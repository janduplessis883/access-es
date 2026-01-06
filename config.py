"""
Configuration file for Access ES Appointment Tracker
Contains all constants, thresholds, and configuration values
"""

import pandas as pd

# Page Configuration
PAGE_TITLE = "Access ES - Appointment Tracker"
PAGE_ICON = ":material/switch_access_shortcut_add:"
PAGE_LAYOUT = "wide"

# Financial Year Dates
FY_START = pd.Timestamp(2025, 4, 1)
FY_END = pd.Timestamp(2026, 3, 31)

# Payment Thresholds
THRESHOLD_100_PERCENT = 85.0  # Apps per 1000 per week for 100% payment
THRESHOLD_75_PERCENT = 75.0   # Apps per 1000 per week for 75% payment

# ARRS Month Mapping (end of month dates)
ARRS_MONTH_MAP = {
    "April 25": pd.Timestamp(2025, 4, 30),
    "May 25": pd.Timestamp(2025, 5, 31),
    "June 25": pd.Timestamp(2025, 6, 30),
    "July 25": pd.Timestamp(2025, 7, 31),
    "August 25": pd.Timestamp(2025, 8, 31),
    "September 25": pd.Timestamp(2025, 9, 30),
    "October 25": pd.Timestamp(2025, 10, 31),
    "November 25": pd.Timestamp(2025, 11, 30),
    "December 25": pd.Timestamp(2025, 12, 31),
    "January 26": pd.Timestamp(2026, 1, 31),
    "February 26": pd.Timestamp(2026, 2, 28),
    "March 26": pd.Timestamp(2026, 3, 31)
}

# Time Constants
WEEKS_PER_YEAR = 52.14
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30.44
WEEKS_PER_MONTH = 4.345

# Plot Configuration
PLOT_COLORS = {
    'actual_appointments': '#0077b6',
    'arrs_historical': '#f48c06',
    'arrs_future': '#cb3f4e',
    'forecasted_apps': '#00b4d8',
    'catchup_needed': '#6a994e',
    'threshold_line': '#c1121f',
    'average_line': '#e4af6c',
    'arrs_cutoff_line': '#749857'
}

# Plot Heights
BASE_PLOT_HEIGHT = 300
HEIGHT_PER_CLINICIAN = 15
BOXPLOT_BASE_HEIGHT = 200
WEEKLY_PLOT_HEIGHT = 500
MONTHLY_PLOT_HEIGHT = 400
PROJECTION_PLOT_HEIGHT = 400

# Required Columns
REQUIRED_COLUMNS = ['appointment_date', 'clinician', 'appointment_status', 'rota_type']

# Column Name Mappings
COLUMN_RENAMES = {
    'appointment_duration_(actual)': 'duration',
    'time_between_booking_and_appointment': 'book_to_app'
}

# Default Values
DEFAULT_LIST_SIZE = 1
DEFAULT_ARRS = 0
SAFE_WEEKS_MIN = 0.1  # Minimum value to prevent division by zero
