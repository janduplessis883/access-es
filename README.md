# Access ES Tracker

A Streamlit dashboard for tracking GP surgery appointments and calculating Access ES metrics for the 2025-26 financial year. Visualize appointment data, exclude "Did Not Attend" cases, and calculate weekly performance rates against targets with optional forecasting.

![screenshot](images/access-es-screen.png)

## Features

- **CSV Data Merging**: Upload multiple appointment CSV files to analyze them as one dataset
- **Interactive Filtering**: Filter by Clinician, Rota Type, and Date Range
- **DNA Exclusion**: Toggle to include or exclude "Did Not Attend" appointments
- **ARRS Handling**: Manually input ARRS (Additional Roles Reimbursement Scheme) figures with double-counting prevention warnings
- **Performance Metrics**: Calculate "Average finished appointments per 1000 patients per week" with visual indicators against the target threshold (> 85 per 1000)
- **Forecasting**: Optional N-BEATS time series forecasting to predict future appointment volumes

---

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package manager)

## Installation

1. Clone or navigate to the repository:
   ```bash
   cd access-es
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Using the Dashboard

### 1. Launch the Application

```bash
streamlit run access-es/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

### 2. Sidebar Settings

Before analyzing data, configure the following settings in the sidebar:

#### File Upload
- Click **"Choose CSV files"** to upload one or more appointment CSV files
- Multiple files will be automatically merged into a single dataset

#### Processing Options
- **Drop Duplicate Entries**: Toggle to remove duplicate appointment records
- **Exclude 'Did Not Attend'**: Toggle to filter out DNA appointments from calculations (recommended: ON)

#### Practice Configuration
- **List Size**: Enter your practice's total patient list size (e.g., 10,000)
- **ARRS 25/26**: Enter the derived monthly ICB figure for ARRS appointments if applicable
  - **Important**: If adding ARRS manually, deselect ARRS clinicians from the Clinician filter to avoid double counting

#### Date Range
- Use the date input to select the analysis period for your report

### 3. Main Dashboard Tabs

The dashboard contains the following tabs:

#### Appointments Overview
- View appointment volume trends over time
- Filter by specific clinicians and rota types
- See appointment status breakdown

#### Performance Metrics
- Calculate the **Average finished appointments per 1000 patients per week**
- Visual performance indicator against the target threshold (> 85 per 1000)
- Payment badge display:
  - 🟢 **100% Payment**: > 85 per 1000 patients/week
  - 🟡 **75% Payment**: 75 - 85 per 1000 patients/week
  - 🟠 **< 75% Payment**: < 75 per 1000 patients/week

#### Forecasting (Optional)
- Enable N-BEATS time series forecasting for appointment prediction
- Uses historical data and optional external covariates (influenza data)
- View validation metrics and future predictions

---

## Data Format Requirements

Upload CSV files with the following columns (headers are case-insensitive):

| Column | Description | Example |
|--------|-------------|---------|
| `appointment_date` | Date of the appointment | `01-Apr-25` or `2025-04-01` |
| `clinician` | Name of the clinician | `Dr. Smith`, `Pharm. Jones` |
| `rota_type` | Type of shift/rota | `GP Clinic`, `Extended Access` |
| `appointment_status` | Status of the appointment | `Finished`, `Did Not Attend` |

---

## Calculations

### Weekly Rate Formula

```
Average = (Total Appointments + ARRS) / List Size * 1000 / Number of Weeks
```

- **Total Appointments**: Count of unique appointments in selected date range (excluding DNA)
- **ARRS**: Manually added figure (optional)
- **Number of Weeks**: Calculated based on the selected date range on the slider (not just where data exists)

### Payment Thresholds

| Threshold | Rate per 1000 Patients/Week |
|-----------|----------------------------|
| 100% Payment | > 85 |
| 75% Payment | 75 - 85 |
| < 75% Payment | < 75 |

---

## Troubleshooting

- **Missing columns**: Ensure your CSV files contain all required columns listed above
- **Double counting**: If using manual ARRS input, verify ARRS clinicians are excluded from the Clinician filter
- **Date format errors**: Use consistent date formats (DD-Mon-YY or YYYY-MM-DD)
