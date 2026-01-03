import streamlit as st
import pandas as pd
import plotly.express as px
import io
from datetime import date

st.set_page_config(page_title="Access ES Tracker", layout="wide", page_icon=':material/child_hat:')

st.title("Access ES Tracker")
st.caption("Check your Access ES appointment provision.")

# Sidebar for file uploads
with st.sidebar:
    st.title(":material/settings: Settings")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Select one or more CSV files to combine"
    )
    drop_duplicates = st.toggle("Drop Duplicate Entries", value=True)
    st.divider()
    st.subheader("Scatter Plot Options")
    plot_view_option = st.radio(
        "Select scatter plot view",
        options=["Rota Type", "App Flags", "DNAs", "Rota/Flags"],
        index=0,
        help="Choose which columns to use for the scatter plot axes and color"
    )


    st.divider()
    
    st.subheader(("'Did Not Attend'"))
    exclude_did_not_attend = st.checkbox(
        "Exclude 'Did Not Attend'",
        value=True,
        help="Filter out appointments with **'Did Not Attend'** status"
    )

    if exclude_did_not_attend:
        st.info("✅ 'Did Not Attend' appointments excluded.")
    else:
        st.error("⚠️ Remove 'Did Not Attend' appointments, for accurate calculations.")
        
    st.divider()
    
    st.subheader("Weighted List Size")   
    list_size = st.number_input(
        "List Size",
        min_value=1,
        value=4000,
        step=1,
        help="Enter your surgery's **weighted list size**, and press **Enter**."
    )   
    st.divider()

    st.header("ARRS Allocation")
    arrs_2526 = st.number_input("Total ARRS to date.", min_value=0, value=0, step=1, help="Enter your **ARRS allocation** for the year do date and press **Enter**, Set the **date of allocation** in the select box below.")

    # Map month selections to end of month dates
    arrs_month_map = {
        "April 25": date(2025, 4, 30),
        "May 25": date(2025, 5, 31),
        "June 25": date(2025, 6, 30),
        "July 25": date(2025, 7, 31),
        "August 25": date(2025, 8, 31),
        "September 25": date(2025, 9, 30),
        "October 25": date(2025, 10, 31),
        "November 25": date(2025, 11, 30),
        "December 25": date(2025, 12, 31),
        "January 26": date(2026, 1, 31),
        "February 26": date(2026, 2, 28),
        "March 26": date(2026, 3, 31)
    }

    if arrs_2526 == 0:
        arrs_future = False
        arrs_month = "April 25"
        arrs_end_date = arrs_month_map[arrs_month]
    else:
        st.error("⚠️ **To prevent double counting:** If entering ARRS data here, ensure ARRS clinicians are **deselected** in the **Rota type** & **Clinician** filters above.")
        arrs_month = st.selectbox("Select month for ARRS input", options=[
            "April 25", "May 25", "June 25", "July 25", "August 25", "September 25", "October 25", "November 25", "December 25", "January 26", "February 26", "March 26"], index=0, help="Select the month corresponding to the ARRS input.")

        arrs_end_date = arrs_month_map[arrs_month]
        arrs_start_date = date(2025, 4, 1)
        arrs_weeks_span = (arrs_end_date - arrs_start_date).days / 7

        arrs_future = st.checkbox('Estimate Future ARRS?', value=False, help="Estimate future ARRS based on current weekly rate")
        if arrs_future:
            estimated_weekly_arrs = arrs_2526 / arrs_weeks_span
            st.info(f"✅ Future ARRS estimation at **{estimated_weekly_arrs:.0f}** ARRS apps per week - or **{estimated_weekly_arrs/list_size*1000:.2f}** ARRS apps per 1000 per week")
        else:
            st.error(f"⚠️ ARRS **will not be applied** to future months! Your Average Apps per 1000 per week will be underestimated.")
    st.divider()

    show_dataframe = st.checkbox(
        "Show Filtered Data Table",
        value=False,
        help="Display the filtered dataframe below the scatter plot"
    )

# Main content area
if uploaded_files:
    # Concatenate all uploaded CSV files
    dataframes = []
    file_names = []

    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            dataframes.append(df)
            file_names.append(uploaded_file.name)
          
        except Exception as e:
            st.sidebar.error(f"✗ Error loading {uploaded_file.name}: {str(e)}")

    if dataframes:
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Process column names: lowercase and replace spaces with underscores
        combined_df.columns = combined_df.columns.str.lower().str.replace(' ', '_')

        # Convert appointment_date column to datetime if it exists
        if 'appointment_date' in combined_df.columns:
            try:
                # Try the expected format first
                combined_df['appointment_date'] = pd.to_datetime(
                    combined_df['appointment_date'],
                    format='%d-%b-%y'
                )
            except ValueError:
                # If that fails, try auto-detection with dayfirst=True for DD-Mon-YY format
                combined_df['appointment_date'] = pd.to_datetime(
                    combined_df['appointment_date'],
                    format='mixed',
                    dayfirst=True
                )
            # Drop rows with invalid dates to prevent issues
            original_rows = len(combined_df)
            combined_df = combined_df.dropna(subset=['appointment_date'])
            if len(combined_df) < original_rows:
                st.sidebar.warning(f"Dropped {original_rows - len(combined_df)} rows with invalid dates")

            # Sort by appointment date
            combined_df = combined_df.sort_values('appointment_date').reset_index(drop=True)

        # Scatter plot section
        st.subheader("Filters and Visualization")
        if drop_duplicates:
            st.badge(f"✅  Dropped **{combined_df.duplicated().sum()}** duplicated rows.", color='blue')
            combined_df = combined_df.drop_duplicates(keep='first')
        else:
            st.badge(f"⚠️ **{combined_df.duplicated().sum()}** Duplicated rows identified.", color='blue')

        # Check if required columns exist
        required_cols = ['appointment_date', 'clinician', 'appointment_status', 'rota_type']
        has_required_cols = all(col in combined_df.columns for col in required_cols)

        if has_required_cols:
            # Get unique clinician and rota_type names
            all_clinicians = sorted(combined_df['clinician'].unique().tolist())
            all_rota_types = sorted(combined_df['rota_type'].unique().tolist())

            # Create columns for multiselects
            col1, col2 = st.columns(2)

            with col1:
                # Multiselect for rota_type (primary filter)
                selected_rota_types = st.multiselect(
                    "Select Rota Types to Display",
                    options=all_rota_types,
                    default=all_rota_types,
                    help="Select which rota types to show"
                )

            # Get clinicians that work in the selected rota types (for default selection)
            if selected_rota_types:
                default_clinicians = sorted(
                    combined_df[combined_df['rota_type'].isin(selected_rota_types)]['clinician'].unique().tolist()
                )
            else:
                default_clinicians = []

            with col2:
                # Multiselect for clinicians (all available, but default to those in selected rota types)
                selected_clinicians = st.multiselect(
                    "Select Clinicians to Display",
                    options=all_clinicians,
                    default=default_clinicians,
                    help="All clinicians are available. Defaults to clinicians in selected rota types"
                )

            if selected_clinicians and selected_rota_types:
                # Filter dataframe by selected clinicians and rota_types
                filtered_df = combined_df[
                    (combined_df['clinician'].isin(selected_clinicians)) &
                    (combined_df['rota_type'].isin(selected_rota_types))
                ]

                # Date range slider
                min_date = filtered_df['appointment_date'].min().date()
                max_date = filtered_df['appointment_date'].max().date()

                # Handle case where min_date == max_date
                if min_date == max_date:
                    date_range = (min_date, max_date)
                else:
                    date_range = st.slider(
                        "Select Date Range",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date),
                        help="Filter appointments by date range"
                    )

                # Filter dataframe by date range
                filtered_df = filtered_df[
                    (filtered_df['appointment_date'].dt.date >= date_range[0]) &
                    (filtered_df['appointment_date'].dt.date <= date_range[1])
                ]

                # Apply exclude_did_not_attend filter BEFORE creating weekly and monthly aggregations
                dna_count_before = len(filtered_df)
                if exclude_did_not_attend:
                    filtered_df = filtered_df[filtered_df['appointment_status'] != 'Did Not Attend']
                    dna_count_excluded = dna_count_before - len(filtered_df)
                    dna_percentage = (dna_count_excluded / dna_count_before) * 100
                    st.success(f"✓ Excluded {dna_count_excluded} 'Did Not Attend' appointments ({dna_percentage:.1f}% of total)")

                # Determine if ARRS should be applied based on slider end date (not data end date)
                # This ensures ARRS is applied if the user selects a date range that extends to the ARRS month end
                slider_end_raw = date_range[1]
                if isinstance(slider_end_raw, int):
                    slider_end_date = date.fromordinal(slider_end_raw)
                else:
                    slider_end_date = date(slider_end_raw.year, slider_end_raw.month, slider_end_raw.day)
                should_apply_arrs = slider_end_date >= arrs_end_date and arrs_2526 > 0

                # Store the actual data end date for time range calculations
                filtered_end_date = filtered_df['appointment_date'].max().date()

                # Create weekly aggregation for all appointments in filtered dataframe
                weekly_df = filtered_df.copy()
                weekly_df['week'] = weekly_df['appointment_date'].dt.to_period('W').dt.start_time
                weekly_agg = weekly_df.groupby('week').size().reset_index(name='total_appointments')

                # Create monthly aggregation for all appointments in filtered dataframe
                monthly_df = filtered_df.copy()
                monthly_df['month'] = monthly_df['appointment_date'].dt.to_period('M').dt.start_time
                monthly_agg = monthly_df.groupby('month').size().reset_index(name='total_appointments')

                # Calculate dynamic height based on number of clinicians
                base_height = 300
                height_per_clinician = 15
                fig_height = base_height + (len(selected_clinicians) * height_per_clinician)

                # Display plot based on selected view option
                if plot_view_option == "Rota Type":
                    # Y: Clinician, Hue: Rota Type
                    fig = px.strip(
                        filtered_df,
                        x='appointment_date',
                        y='clinician',
                        color='rota_type',
                        title="Appointments by Date and Clinician (Colored by Rota Type)",
                        labels={
                            'appointment_date': 'Appointment Date',
                            'clinician': 'Clinician',
                            'rota_type': 'Rota Type'
                        },
                        height=fig_height
                    )

                elif plot_view_option == "App Flags":
                    # Y: Clinician, Hue: Appointment Flags
                    fig = px.strip(
                        filtered_df,
                        x='appointment_date',
                        y='clinician',
                        color='appointment_flags',
                        title="Appointments by Date and Clinician (Colored by Appointment Flags)",
                        labels={
                            'appointment_date': 'Appointment Date',
                            'clinician': 'Clinician',
                            'appointment_flags': 'Appointment Flags'
                        },
                        height=fig_height
                    )

                elif plot_view_option == "DNAs":
                    # Y: Clinician, Hue: DNAs (appointment_status)
                    # Sort so 'Finished' appointments are plotted first, 'Did Not Attend' on top
                    plot_df_dna = filtered_df.copy()
                    status_order = {'Finished': 0, 'Did Not Attend': 1}
                    plot_df_dna['_status_sort'] = plot_df_dna['appointment_status'].map(status_order)
                    plot_df_dna = plot_df_dna.sort_values('_status_sort')

                    fig = px.strip(
                        plot_df_dna,
                        x='appointment_date',
                        y='clinician',
                        color='appointment_status',
                        title="Appointments by Date and Clinician (Colored by DNAs)",
                        labels={
                            'appointment_date': 'Appointment Date',
                            'clinician': 'Clinician',
                            'appointment_status': 'DNAs'
                        },
                        height=fig_height
                    )
                    # Set marker opacity for transparency
                    fig.update_traces(marker=dict(opacity=0.6))

                elif plot_view_option == "Rota/Flags":
                    # Y: Appointment Flags, Hue: Rota Type
                    num_flags = filtered_df['appointment_flags'].nunique()
                    flags_base_height = 300
                    height_per_flag = 15
                    flags_fig_height = flags_base_height + (num_flags * height_per_flag)

                    fig = px.strip(
                        filtered_df,
                        x='appointment_date',
                        y='appointment_flags',
                        color='rota_type',
                        title="Appointments by Date and Flags (Colored by Rota Type)",
                        labels={
                            'appointment_date': 'Appointment Date',
                            'appointment_flags': 'Appointment Flags',
                            'rota_type': 'Rota Type'
                        },
                        height=flags_fig_height
                    )
                    fig_height = flags_fig_height

                fig.update_layout(
                    xaxis_title="Appointment Date",
                    yaxis_title="Clinician" if plot_view_option != "Rota/Flags" else "Appointment Flags",
                    hovermode='closest',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                )

                st.plotly_chart(fig, width='stretch')

                # Calculate threshold line
                threshold = (85 / 1000) * list_size

                # Calculate average weekly appointments
                avg_weekly_appointments = weekly_agg['total_appointments'].mean()

                # Create line plot for weekly appointments
                fig_weekly = px.line(
                    weekly_agg,
                    x='week',
                    y='total_appointments',
                    title='Total Appointments per Week',
                    labels={
                        'week': 'Week Starting Date',
                        'total_appointments': 'Total Appointments'
                    },
                    height=400,
                    markers=True
                )

                # Add average line to weekly plot
                fig_weekly.add_hline(
                    y=avg_weekly_appointments,
                    line_dash='dot',
                    line_color='#e4af6c',
                    annotation_text=f'Weekly Average Completed Apps ({avg_weekly_appointments:.1f})',
                    annotation_position='top left'
                )

                # Add threshold line to weekly plot
                fig_weekly.add_hline(
                    y=threshold,
                    line_dash='dash',
                    line_color='#ae4f4d',
                    annotation_text=f'Threshold ({threshold:.2f})',
                    annotation_position='top right'
                )

                # Add ARRS end date vertical line if applicable
                if should_apply_arrs:
                    fig_weekly.add_vline(
                        x=pd.Timestamp(arrs_end_date),
                        line_dash='dashdot',
                        line_color='#749857'
                    )
                    fig_weekly.add_annotation(
                        x=pd.Timestamp(arrs_end_date),
                        text='ARRS Prediction Start',
                        showarrow=False,
                        xanchor='right',
                        yanchor='top'
                    )

                # Add estimated ARRS indicator if applicable
                if arrs_future and should_apply_arrs:
                    # Add yellow horizontal band showing estimated weekly ARRS rate
                    fig_weekly.add_hline(
                        y=estimated_weekly_arrs,
                        line_dash='dot',
                        line_color='#7d2e61',
                        annotation_text=f'Est. Weekly ARRS ({estimated_weekly_arrs:.0f})',
                        annotation_position='top left'
                    )

                fig_weekly.update_layout(
                    xaxis_title="Week Starting Date",
                    yaxis_title="Total Appointments",
                    hovermode='x unified',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                )

                st.plotly_chart(fig_weekly, width='stretch')

                # Calculate average monthly appointments
                avg_monthly_appointments = monthly_agg['total_appointments'].mean()

                # Create line plot for monthly appointments without threshold line
                fig_monthly = px.line(
                    monthly_agg,
                    x='month',
                    y='total_appointments',
                    title='Total Appointments per Month',
                    labels={
                        'month': 'Month',
                        'total_appointments': 'Total Appointments'
                    },
                    height=400,
                    markers=True
                )

                # Add average line to monthly plot
                fig_monthly.add_hline(
                    y=avg_monthly_appointments,
                    line_dash='dot',
                    line_color='#e4af6c',
                    annotation_text=f'Monthly Average Completed Apps ({avg_monthly_appointments:.1f})',
                    annotation_position='top left'
                )

                # Add ARRS end date vertical line if applicable
                if should_apply_arrs:
                    fig_monthly.add_vline(
                        x=pd.Timestamp(arrs_end_date),
                        line_dash='dashdot',
                        line_color='#749857'
                    )
                    fig_monthly.add_annotation(
                        x=pd.Timestamp(arrs_end_date),
                        text='ARRS Prediction Start',
                        showarrow=False,
                        xanchor='right',
                        yanchor='top'
                    )

                # Add estimated ARRS indicator if applicable
                if arrs_future and should_apply_arrs:
                    # Add yellow horizontal band showing estimated weekly ARRS rate
                    fig_monthly.add_hline(
                        y=estimated_weekly_arrs,
                        line_dash='dot',
                        line_color='#7d2e61',
                        annotation_text=f'Est. Weekly ARRS ({estimated_weekly_arrs:.0f})',
                        annotation_position='top left'
                    )

                fig_monthly.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Total Appointments",
                    hovermode='x unified',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                )

                st.plotly_chart(fig_monthly, width='stretch')

                # Display filtered dataframe if checkbox is enabled
                if show_dataframe:
                    st.subheader("Filtered Data Table")
                    st.dataframe(filtered_df, width='stretch')

                # Display dynamic statistics based on filtered data
                st.subheader("Data Statistics")
                col1, col2 = st.columns([1,6])
                with col1:
                    if drop_duplicates:
                        st.badge("✅ Duplicates dropped", color='green')
                    else:
                        st.badge("⚠️ Duplicate entries", color='yellow')
                with col2:
                    if exclude_did_not_attend:
                        st.badge("✅ Excluding 'Did Not Attend'", color='green')
                    else:
                        st.badge("⚠️ Includes 'Did Not Attend'", color='yellow')  
                    
                # Calculate time_diff based on SLIDER range, not data range
                # This ensures the denominator (weeks) correctly reflects the period selected by the user,
                # even if there are gaps in the data (e.g. no appointments on weekends or holidays).
                time_diff = (date_range[1] - date_range[0]).days
                weeks = time_diff / 7
                months = time_diff / 30.44

                if arrs_future and should_apply_arrs:
                    # Calculate weeks from ARRS end date to slider end date for future estimation
                    future_weeks = (slider_end_date - arrs_end_date).days / 7
                    future_arrs_apps = int(round(estimated_weekly_arrs * future_weeks, 0)) if future_weeks > 0 else 0
                    total_apps_arrs = len(filtered_df) + arrs_2526 + future_arrs_apps
                elif should_apply_arrs:
                    total_apps_arrs = len(filtered_df) + arrs_2526
                    future_arrs_apps = 0
                else:
                    total_apps_arrs = len(filtered_df)
                    future_arrs_apps = 0

                av_1000_week = (total_apps_arrs / list_size) * 1000 / weeks if weeks > 0 else 0


                col1, col2, col3, col4 = st.columns(4)
                start_date = filtered_df['appointment_date'].min().strftime('%d %b %y')
                end_date = filtered_df['appointment_date'].max().strftime('%d %b %y')
                with col1:
                    st.metric("Total Surgery Appointments", len(filtered_df))
                    st.badge(f"{len(filtered_df)/13958*1000/weeks:.2f} apps per 1000 per week")

                with col2:
                    if should_apply_arrs:
                        if arrs_future:
                            total_arrs_estimated = arrs_2526 + future_arrs_apps
                            st.metric(f"Total ARRS estimated to {end_date}", total_arrs_estimated)
                            st.badge(f"+ Future ARRS Applied! ({arrs_2526} + {future_arrs_apps})", color='yellow')
                        else:
                            st.metric(f"Total ARRS to end {arrs_month}", arrs_2526)
                            st.badge(f"ARRS Applied till {arrs_month}", color='green')
                    else:
                        st.metric("Total ARRS Applied", 0)
                        st.badge(f"No ARRS - End date before {arrs_month}", color='orange')

                with col3:
                    st.metric("Start Date", start_date)
                with col4:
                    st.metric("End Date", end_date)


                # Second row of metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Time Range (Weeks)", f"{weeks:.1f}")
                with col2:
                    st.metric("Time Range (Months)", f"{months:.1f}")
                with col3:
                    st.metric("Total apps + ARRS", f"{total_apps_arrs}")
                with col4:
                    st.metric("Average apps per 1000 per week", f"{av_1000_week:.2f}")
                    if av_1000_week > 200:
                        st.badge(f"Enter Weighted list size in sidebar", color='red')
                    elif av_1000_week > 85:
                        st.badge(f"100% Access Payment", color='green')
                    elif av_1000_week >= 75 and av_1000_week <= 85:
                        st.badge(f"75% Access Payment", color='yellow')
                    else:
                        st.badge(f"< 75% Access Payment", color='orange')



            else:
                st.warning("Please select at least one clinician and rota type to display.")
        else:
            st.warning("Required columns (appointment_date, clinician, appointment_status, rota_type) not found in the uploaded data.")



        # Show file information
        with st.expander("File & Debug Information", icon=":material/info:"):
            # ===== DATA STATISTICS SECTION =====
            st.markdown("### :material/info: Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.badge(f":material/folder: Original Rows: {len(combined_df)}", color="blue")
            with col2:
                if 'filtered_df' in locals():
                    st.badge(f":material/search: Filtered Rows: {len(filtered_df)}", color="violet")
                else:
                    st.badge(f":material/search: Filtered Rows: N/A", color="gray")
            with col3:
                st.badge(f":material/table: Columns: {len(combined_df.columns)}", color="blue")
            with col4:
                st.badge(f":material/folder: Files Loaded: {len(file_names)}", color="blue")
            
            # File details with dates
            st.markdown("**File Details:**")
            total_loaded = 0
            file_cols = st.columns(min(3, len(file_names)) if len(file_names) > 0 else 1)
            for i, name in enumerate(file_names):
                df_rows = len(dataframes[i])
                total_loaded += df_rows
                # Get date range for this file
                if 'appointment_date' in dataframes[i].columns:
                    file_min = pd.to_datetime(dataframes[i]['appointment_date'], format='%d-%b-%y', errors='coerce').min()
                    file_max = pd.to_datetime(dataframes[i]['appointment_date'], format='%d-%b-%y', errors='coerce').max()
                    if pd.notna(file_min) and pd.notna(file_max):
                        date_range_str = f"{file_min.strftime('%d %b %y')} → {file_max.strftime('%d %b %y')}"
                        with file_cols[i % len(file_cols)]:
                            st.badge(f"'{name}': {df_rows} rows | {date_range_str}", color="blue")
                    else:
                        with file_cols[i % len(file_cols)]:
                            st.badge(f"'{name}': {df_rows} rows | :material/error: Date parsing failed", color="orange")
                else:
                    with file_cols[i % len(file_cols)]:
                        st.badge(f"'{name}': {df_rows} rows", color="blue")
            
            col1, col2 = st.columns(2)
            with col1:
                st.badge(f":material/check_circle: Total Loaded: {total_loaded}", color="green")
            with col2:
                st.badge(f":material/check_circle: After Cleaning: {len(combined_df)}", color="green")
            
            st.divider()
            
            # ===== INPUT CONFIGURATION SECTION =====
            st.markdown("### :material/settings: Input Configuration")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.badge(f":material/menu: List Size: {list_size}", color="green")
            with col2:
                st.badge(f":material/star: ARRS Entered: {arrs_2526}", color="green")
            with col3:
                status_text = ":material/check: Yes" if arrs_future else ":material/close: No"
                st.badge(f":material/star: Future ARRS: {status_text}", color="green")
            with col4:
                status_text = ":material/check: Yes" if exclude_did_not_attend else ":material/close: No"
                st.badge(f":material/close: Exclude DNAs: {status_text}", color="green")
            
            st.divider()
            
            # ===== DATE RANGE SECTION =====
            st.markdown("### :material/calendar_check: Date Range")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.badge(f":material/play_arrow: Slider Start: {date_range[0]}", color="orange")
            with col2:
                st.badge(f":material/close: Slider End: {slider_end_date}", color="orange")
            with col3:
                st.badge(f":material/calendar_check: Data Start: {filtered_df['appointment_date'].min().date()}", color="orange")
            with col4:
                st.badge(f":material/calendar_check: Data End: {filtered_end_date}", color="orange")
            
            st.divider()
            
            # ===== APPOINTMENT COUNTS SECTION =====
            st.markdown("### :material/call: Appointment Counts")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.badge(f":material/search: Filtered Apps: {len(filtered_df)}", color="blue")
            with col2:
                st.badge(f":material/star: ARRS Applied: {arrs_2526}", color="blue")
            with col3:
                st.badge(f":material/star: Future ARRS Est: {future_arrs_apps}", color="blue")
            with col4:
                st.badge(f":material/add: Total + ARRS: {total_apps_arrs}", color="blue")
            
            st.divider()
            
            # ===== TIME CALCULATION SECTION =====
            st.markdown("### :material/schedule: Time Calculations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.badge(f":material/calendar_check: Days in Range: {time_diff}", color="violet")
            with col2:
                st.badge(f":material/calendar_check: Weeks: {weeks:.2f}", color="violet")
            with col3:
                st.badge(f":material/calendar_check: Months: {months:.1f}", color="violet")
            
            st.divider()
            
            # ===== FINAL CALCULATION SECTION =====
            st.markdown("### :material/edit: Final Calculation")
            st.badge(f"Formula: ({total_apps_arrs} ÷ {list_size}) × 1000 ÷ {weeks:.2f} = {av_1000_week:.2f} apps per 1000 per week", color="red")
            
            # Visual breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.badge(f":material/add: Total Apps: {total_apps_arrs}", color="red")
            with col2:
                st.badge(f":material/menu: List Size: {list_size}", color="red")
            with col3:
                st.badge(f":material/edit: Result: {av_1000_week:.2f}", color="red")

        # Download combined CSV
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Combined CSV",
            data=csv,
            file_name="combined_data.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Please upload CSV files using the sidebar to get started!")
