# Access ES Tracker - Current Implementation State

## Overview
This is a Streamlit application for tracking appointments and calculating Access metrics for the 2025-26 financial year. The app merges CSV files, applies filters, and calculates key performance indicators related to appointment scheduling and ARRS (Appointments and Related Services).

---

## Key Architecture & Data Flow

### 1. **Data Loading & Processing**
- Multiple CSV files are uploaded and concatenated
- Column names are normalized to lowercase with underscores replacing spaces
- `appointment_date` is converted to datetime with flexible format detection:
  - First tries format: `'%d-%b-%y'` (DD-Mon-YY, e.g., "01-Apr-25")
  - If that fails, auto-detects format using `format='mixed'` with `dayfirst=False`
  - This supports both traditional format and ISO format (YYYY-MM-DD)

### 2. **Filtering Pipeline** (Applied in Order)
1. **Clinician Filter**: User selects clinicians via multiselect
   - Default: All clinicians in selected rota types
2. **Rota Type Filter**: User selects rota types via multiselect
   - Default: All rota types available
3. **Date Range Filter**: User adjusts slider for start/end dates
   - Range: From min to max appointment dates in data
4. **DNA Exclusion Filter**: Checkbox "Exclude 'Did Not Attend'" (default=True)
   - When TRUE: Removes all DNA appointments from `filtered_df`
   - Affects ALL subsequent calculations, charts, and metrics

---

## Critical Calculation Logic

### A. **ARRS Date Handling**

#### ARRS Month Mapping
Maps user-selected months to END-OF-MONTH dates:
```python
arrs_month_map = {
    "April 25": date(2025, 4, 30),      # End of April
    "May 25": date(2025, 5, 31),        # End of May
    "June 25": date(2025, 6, 30),       # End of June
    "July 25": date(2025, 7, 31),       # End of July
    "August 25": date(2025, 8, 31),     # End of August
    "September 25": date(2025, 9, 30),  # End of September
    "October 25": date(2025, 10, 31),   # End of October
    "November 25": date(2025, 11, 30),  # End of November
    "December 25": date(2025, 12, 31),  # End of December
    "January 26": date(2026, 1, 31),    # End of January
    "February 26": date(2026, 2, 28),   # End of February
    "March 26": date(2026, 3, 31)       # End of March
}
```

#### ARRS Weeks Span Calculation
- **Start Date**: Always April 1, 2025
- **End Date**: Selected month's last day
- **Formula**: `(end_date - start_date).days / 7`
- **Example**: April 25 = (April 30 - April 1) / 7 = 4.14 weeks

#### ARRS Application Conditions
ARRS is ONLY applied if BOTH conditions are met:
```python
should_apply_arrs = filtered_end_date >= arrs_end_date AND arrs_2526 > 0
```
- `filtered_end_date`: Last appointment date in filtered data
- `arrs_end_date`: End of selected month
- If filtered data ends BEFORE ARRS end date → ARRS NOT applied

### B. **Future ARRS Estimation**

When "Estimate Future ARRS?" checkbox is TRUE:
```
estimated_weekly_arrs = arrs_2526 / arrs_weeks_span
future_weeks = (filtered_end_date - arrs_end_date).days / 7
future_arrs_apps = estimated_weekly_arrs * future_weeks (rounded to int)
```

#### Total Apps + ARRS Calculation (3 Scenarios)

**Scenario 1**: Future ARRS estimation enabled AND ARRS should apply
```
total_apps_arrs = len(filtered_df) + arrs_2526 + future_arrs_apps
```

**Scenario 2**: Future ARRS OFF AND ARRS should apply
```
total_apps_arrs = len(filtered_df) + arrs_2526
```

**Scenario 3**: ARRS should NOT apply (date before ARRS end date)
```
total_apps_arrs = len(filtered_df)
```

### C. **Time Range Calculations**

All use `filtered_df` appointments (DNA-excluded if checkbox enabled):

```python
time_diff = (max_appointment_date - min_appointment_date).days

weeks = time_diff / 7                    # "Time Range (Weeks)"
months = time_diff / 30.44               # "Time Range (Months)" - Using 30.44 avg days per month
```

### D. **Average Apps per 1000 per Week Calculation**

```python
av_1000_week = (total_apps_arrs / list_size) * 1000 / weeks

if weeks > 0:
    av_1000_week = (total_apps_arrs / list_size) * 1000 / weeks
else:
    av_1000_week = 0
```

**Important**:
- Uses `total_apps_arrs` (includes ARRS when applicable)
- Divided by full `weeks` in filtered range (even if ARRS doesn't apply)
- Uses user-entered `list_size` from sidebar

### E. **Threshold Line (Weekly Chart)**

```python
threshold = (85 / 1000) * list_size
```

- Represents the weekly appointment count corresponding to 85 appointments per 1000 list size
- Example: list_size=100 → threshold=8.5 per week

### F. **Payment Status Badges**

Based on `av_1000_week`:
- **> 200**: Red badge "Enter Weighted list size in sidebar"
- **85 to 200** (inclusive): Green badge "100% Access Payment"
- **75 to <85**: Yellow badge "75% Access Payment"
- **< 75**: Orange badge "< 75% Access Payment"

---

## DNA Exclusion Filter

**Checkbox**: "Exclude 'Did Not Attend'" (default=TRUE)

**Current Logic** (CORRECT):
- When checkbox is TRUE: Filter removes DNA appointments
- When checkbox is FALSE: DNA appointments are INCLUDED in calculations

**Applied at**:
```python
if exclude_did_not_attend:
    filtered_df = filtered_df[filtered_df['appointment_status'] != 'Did Not Attend']
```

**Affects**:
- Total Surgery Appointments count
- Weekly/Monthly aggregations
- All charts (scatter, line plots)
- All metrics including ARRS calculations

---

## Metrics Display Structure

### First Row (4 metrics):
1. **Total Surgery Appointments**: `len(filtered_df)` (DNA-excluded)
2. **Total ARRS Applied**: Shows ARRS status with conditional badges
   - If future estimation: Shows breakdown (base + estimated)
   - If normal: Shows base ARRS only
   - If not applicable: Shows 0
3. **Start Date**: First appointment in filtered range
4. **End Date**: Last appointment in filtered range

### Second Row (4 metrics):
1. **Time Range (Weeks)**: `weeks` value
2. **Time Range (Months)**: `months` value
3. **Total apps + ARRS**: `total_apps_arrs` value
4. **Average apps per 1000 per week**: `av_1000_week` with payment status badge

---

## Critical Notes for Future Development

### ⚠️ Important Constraints
1. ARRS always starts from April 1, 2025 - do NOT change this
2. End-of-month dates are used for ARRS mapping - verify this is correct
3. DNA exclusion must be applied BEFORE creating weekly/monthly aggregations
4. The `filtered_df` state after DNA exclusion is used for ALL subsequent calculations
5. Weeks denominator is ALWAYS the full filtered range, regardless of ARRS date bounds

### 🔧 If You Need to Change Logic
- **Weeks Calculation**: Update `time_diff / 7` consistently everywhere
- **ARRS Dates**: Update `arrs_month_map` dictionary
- **ARRS Application**: Modify `should_apply_arrs` condition carefully
- **Future ARRS**: Verify `future_weeks` calculation if changing date logic
- **DNA Filter**: Update the conditional `if exclude_did_not_attend:` block

### 📊 If You Need to Add New Metrics
- Ensure they use `filtered_df` (with DNA exclusion applied)
- If using time-based rates: Divide by `weeks` consistently
- If including ARRS: Use `total_apps_arrs` not `len(filtered_df)`

### 🧪 Testing scenarios to verify
1. ✓ DNA checkbox ON/OFF changes all metric values
2. ✓ Date slider before ARRS end date → ARRS not applied
3. ✓ Date slider after ARRS end date → ARRS applied
4. ✓ Future ARRS estimation adds to base ARRS in metrics
5. ✓ Payment status badges change based on av_1000_week threshold

---

## File Structure
- **app.py**: Main application file with all logic
- **requirements.txt**: Python dependencies
- **.streamlit/config.toml**: Streamlit configuration
- **current_state.md**: This file

---

## Last Updated
February 1, 2026 - All calculations verified and corrected. ARRS logic finalized with end-of-month date mapping.
