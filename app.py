"""
Access ES - Appointment Tracker
Refactored version using modular architecture
"""

import streamlit as st
import pandas as pd
import io
from config import *
from utils import *
from plots import *

# Page Configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    page_icon=PAGE_ICON
)

# Header
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(
    "Check your **:material/switch_access_shortcut_add: Access ES** appointment provision. "
    "To ensure accurate results complete all the **settings** in the **sidebar**. "
    "Especially ARRS up to the date this value has been supplied by the ICB."
)
st.logo("logo.png", size="small")

# Sidebar Configuration
with st.sidebar:
    st.title(":material/settings: Settings")
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Select one or more CSV files to combine"
    )
    
    # Processing Options
    drop_duplicates = st.toggle("Drop Duplicate Entries", value=True)
    exclude_did_not_attend = st.toggle(
        "Exclude 'Did Not Attend'",
        value=True,
        help="Filter out appointments with **'Did Not Attend'** status"
    )
    
    if exclude_did_not_attend:
        st.info(":material/done_outline: 'Did Not Attend' appointments excluded.")
    else:
        st.error(":material/warning: Remove 'Did Not Attend' appointments, for accurate calculations.")
    
    st.divider()
    
    # Scatter Plot Options
    st.subheader("Scatter Plot Options")
    plot_view_option = st.radio(
        "Select scatter plot view",
        options=["Rota Type", "App Flags", "DNAs", "Rota/Flags"],
        index=0,
        help="Choose which columns to use for the scatter plot axes and color"
    )
    
    st.divider()
    
    # Weighted List Size
    st.subheader("Weighted List Size")
    list_size = st.number_input(
        "Weighted List Size",
        min_value=1,
        value=DEFAULT_LIST_SIZE,
        step=1,
        help="Enter your surgery's **weighted list size**, and press **Enter**."
    )
    
    st.divider()
    
    # ARRS Configuration
    st.header("ARRS Allocation")
    arrs_2526 = st.number_input(
        "Total ARRS to date.",
        min_value=0,
        value=DEFAULT_ARRS,
        step=1,
        help="Enter your **ARRS allocation** for the year to date and press **Enter**. "
             "Set the **date of allocation** in the select box below. Enter 0 if no ARRS data."
    )
    
    if arrs_2526 == 0:
        arrs_future = False
        arrs_month = "April 25"
        arrs_end_date = ARRS_MONTH_MAP[arrs_month]
        estimated_weekly_arrs = 0.0
    else:
        st.error(
            ":material/warning: **To prevent double counting:** If entering ARRS data here, "
            "ensure ARRS clinicians are **deselected** in the **Rota type** & **Clinician** filters above."
        )
        arrs_month = st.selectbox(
            "Select month for ARRS input",
            options=list(ARRS_MONTH_MAP.keys()),
            index=0,
            help="Select the month corresponding to the ARRS input."
        )
        
        arrs_end_date = ARRS_MONTH_MAP[arrs_month]
        arrs_start_date = FY_START
        arrs_weeks_span = max(0.1, (arrs_end_date - arrs_start_date).days / DAYS_PER_WEEK)
        estimated_weekly_arrs = arrs_2526 / arrs_weeks_span
        
        arrs_future = st.toggle(
            'Apply estimated future ARRS?',
            value=False,
            help="Estimate future ARRS based on current weekly rate"
        )
        
        if arrs_future:
            st.info(
                f":material/done_outline: Future ARRS estimation at **{estimated_weekly_arrs:.0f}** "
                f"ARRS apps per week - or **{estimated_weekly_arrs/list_size*1000:.2f}** ARRS apps per 1000 per week"
            )
        else:
            st.error(
                ":material/warning: ARRS **will not be applied** to future months! "
                "Your Average Apps per 1000 per week will be underestimated."
            )
    
    st.divider()
    
    # Experimental Features
    st.subheader(":material/labs: Experimental")
    exp_add_apps_per_week = st.slider(
        "Experiment: Add Apps per Week",
        min_value=0,
        max_value=500,
        value=0,
        step=5,
        help="Simulate adding more appointments per week for the REMAINING period of the financial year."
    )
    
    exp_add_total_apps = st.slider(
        "Experiment: Add Total Apps",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
        help="Simulate adding total appointments to the historical data. "
             "This affects the Average apps per 1000 per week calculation."
    )
    
    st.divider()
    
    # Display Options
    show_dataframe = st.checkbox(
        ":material/table: Show DataFrame",
        value=False,
        help="Display the filtered dataframe below the scatter plot"
    )

# Main Content Area
if uploaded_files:
    # Load and Combine Files
    combined_df, file_info = load_and_combine_csv_files(uploaded_files)
    
    # Display file loading errors
    for info in file_info:
        if not info['success']:
            st.sidebar.error(f":material/close: Error loading {info['name']}: {info['error']}")
    
    if combined_df is not None:
        # Preprocess Data
        combined_df, rows_dropped = preprocess_dataframe(combined_df)
        
        if rows_dropped > 0:
            st.sidebar.warning(f"Dropped {rows_dropped} rows with invalid dates")
        
        # Filters and Visualization Section
        st.subheader("Filters and Visualization")
        
        # Duplicate handling (badge will be shown after date slider)
        dup_count = combined_df.duplicated().sum()
        if drop_duplicates:
            combined_df = combined_df.drop_duplicates(keep='first')
        
        # Check required columns
        has_required_cols = all(col in combined_df.columns for col in REQUIRED_COLUMNS)
        
        if has_required_cols:
            # Get unique values for filters
            all_clinicians = sorted(combined_df['clinician'].unique().tolist())
            all_rota_types = sorted(combined_df['rota_type'].unique().tolist())
            
            # Create filter columns
            col1, col2 = st.columns(2)
            
            with col1:
                selected_rota_types = st.multiselect(
                    "Select Rota Types to Display",
                    options=all_rota_types,
                    default=all_rota_types,
                    help="Select which rota types to show"
                )
            
            # Default clinicians based on selected rota types
            if selected_rota_types:
                default_clinicians = sorted(
                    combined_df[combined_df['rota_type'].isin(selected_rota_types)]['clinician'].unique().tolist()
                )
            else:
                default_clinicians = []
            
            with col2:
                selected_clinicians = st.multiselect(
                    "Select Clinicians to Display",
                    options=all_clinicians,
                    default=default_clinicians,
                    help="All clinicians are available. Defaults to clinicians in selected rota types"
                )
            
            if selected_clinicians and selected_rota_types:
                # Apply filters
                filtered_df, filtered_df_with_dna, dna_count, dna_percentage = filter_dataframe(
                    combined_df,
                    selected_clinicians,
                    selected_rota_types,
                    date_range=None,  # Will be set after slider
                    exclude_dna=False  # Will apply later
                )
                
                # Date Range Slider
                min_date = filtered_df['appointment_date'].min().date()
                max_date = filtered_df['appointment_date'].max().date()
                
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
                
                # Apply date filter
                filtered_df = filtered_df[
                    (filtered_df['appointment_date'].dt.date >= date_range[0]) &
                    (filtered_df['appointment_date'].dt.date <= date_range[1])
                ]
                filtered_df_with_dna = filtered_df_with_dna[
                    (filtered_df_with_dna['appointment_date'].dt.date >= date_range[0]) &
                    (filtered_df_with_dna['appointment_date'].dt.date <= date_range[1])
                ]
                
                # Show duplicate handling badge
                if drop_duplicates:
                    st.badge(f":material/done_outline: Dropped **{dup_count}** duplicate rows.", color='blue')
                else:
                    st.badge(f":material/warning: **{dup_count}** Duplicate rows identified.", color='yellow')
                
                # Apply DNA exclusion
                dna_count_before = len(filtered_df)
                if exclude_did_not_attend:
                    filtered_df = filtered_df[filtered_df['appointment_status'] != 'Did Not Attend']
                    dna_count = dna_count_before - len(filtered_df)
                    dna_percentage = (dna_count / dna_count_before * 100) if dna_count_before > 0 else 0
                    st.badge(
                        f":material/done_outline: Excluded {dna_count} 'Did Not Attend' appointments "
                        f"({dna_percentage:.1f}% of total)",
                        color='blue'
                    )
                
                # Calculate time metrics
                time_metrics = calculate_time_metrics(date_range)
                
                # Calculate ARRS values
                slider_end_date = date_range[1]
                filtered_end_date = filtered_df['appointment_date'].max()
                
                arrs_values = calculate_arrs_values(
                    arrs_2526,
                    arrs_month,
                    arrs_future,
                    slider_end_date,
                    arrs_end_date,
                    time_metrics['weeks']
                )
                arrs_values['arrs_2526'] = arrs_2526
                arrs_values['arrs_end_date'] = arrs_end_date
                arrs_values['arrs_future'] = arrs_future
                
                # Calculate total appointments
                total_surgery_apps = len(filtered_df) + exp_add_total_apps
                
                if arrs_values['arrs_future'] and arrs_values['should_apply_arrs']:
                    total_apps_arrs = total_surgery_apps + arrs_2526 + arrs_values['future_arrs_apps']
                elif arrs_values['should_apply_arrs']:
                    total_apps_arrs = total_surgery_apps + arrs_2526
                else:
                    total_apps_arrs = total_surgery_apps
                
                av_1000_week = calculate_apps_per_1000(
                    total_apps_arrs,
                    list_size,
                    time_metrics['safe_weeks']
                )
                
                # Create aggregations
                cutoff_date = ARRS_MONTH_MAP[arrs_month]
                
                weekly_agg = create_weekly_aggregation(
                    filtered_df,
                    list_size,
                    arrs_values,
                    cutoff_date
                )
                
                monthly_agg = create_monthly_aggregation(
                    filtered_df,
                    list_size,
                    arrs_values,
                    cutoff_date
                )
                
                # Scatter Plot Visualization
                with st.expander("Visualizations - Appointments Scatter Plot", icon=":material/scatter_plot:", expanded=False):
                    st.subheader(":material/scatter_plot: Visualizations")
                    
                    fig = create_scatter_plot(filtered_df, plot_view_option, selected_clinicians)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Duration statistics
                    st.info(
                        f"Average Appointment length: **{filtered_df['duration'].mean():.2f} minutes** "
                        f"(Across all clinics selected above) Max: {filtered_df['duration'].max():.2f} minutes "
                        f"Min: {filtered_df['duration'].min():.2f} minutes"
                    )
                    st.info(
                        f"Average Time from Booking till Appointment: **{filtered_df['book_to_app'].mean():.2f} minutes** "
                        f"(Across all clinics selected above) Max: {filtered_df['book_to_app'].max():.2f} minutes "
                        f"Min: {filtered_df['book_to_app'].min():.2f} minutes"
                    )
                
                # Weekly & Monthly Trends
                with st.expander("Visualizations - Weekly & Monthly Trends", icon=":material/bar_chart:", expanded=False):
                    st.subheader(":material/bar_chart: Weekly & Monthly Trends")
                    
                    # Toggle for weekly view
                    show_per_1000 = st.toggle(
                        "Apps per 1000 per week",
                        value=False,
                        help="Toggle between Actual Appointment counts (Bar) and Apps per 1000 per week (Line)"
                    )
                    
                    # Weekly plot
                    fig_weekly = create_weekly_trend_plot(
                        weekly_agg,
                        show_per_1000,
                        list_size,
                        arrs_end_date,
                        arrs_values['should_apply_arrs']
                    )
                    st.plotly_chart(fig_weekly, width='stretch')
                    
                    # Monthly plot
                    fig_monthly = create_monthly_trend_plot(
                        monthly_agg,
                        arrs_end_date,
                        arrs_values['should_apply_arrs']
                    )
                    st.plotly_chart(fig_monthly, width='stretch')
                
                # Data Statistics
                expander_open = not show_dataframe
                
                with st.expander("**Data Statistics**", icon=":material/database:", expanded=expander_open):
                    st.subheader(":material/database: Data Statistics")
                    
                    col1, col2 = st.columns([1, 6])
                    with col1:
                        if drop_duplicates:
                            st.badge(":material/done_outline: Duplicates dropped", color='green', help=":material/done_outline: Duplicated can be left but try with and without.")
                        else:
                            st.badge(":material/warning: Duplicate entries", color='yellow')
                    
                    with col2:
                        if exclude_did_not_attend:
                            st.badge(":material/done_outline: Excluding 'Did Not Attend'", color='green')
                        else:
                            st.badge(":material/warning: Includes 'Did Not Attend'", color='yellow')
                    
                    # First row metrics
                    col1, col2, col3, col4 = st.columns(4)
                    start_date = filtered_df['appointment_date'].min().strftime('%d %b %y')
                    end_date = filtered_df['appointment_date'].max().strftime('%d %b %y')
                    
                    with col1:
                        st.metric("Total Surgery Appointments", total_surgery_apps)
                        st.badge(
                            f"{total_surgery_apps/list_size*1000/time_metrics['safe_weeks']:.2f} apps per 1000 per week"
                        )
                    
                    with col2:
                        if arrs_values['should_apply_arrs']:
                            if arrs_future:
                                total_arrs_estimated = arrs_2526 + arrs_values['future_arrs_apps']
                                st.metric(f"Total ARRS estimated to {end_date}", total_arrs_estimated)
                                st.badge(
                                    f"+ Future ARRS Applied! ({arrs_2526} + {arrs_values['future_arrs_apps']})",
                                    color='yellow'
                                )
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
                    
                    # Second row metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Time Range (Weeks)", f"{time_metrics['weeks']:.1f}")
                    with col2:
                        st.metric("Time Range (Months)", f"{time_metrics['months']:.1f}")
                    with col3:
                        st.metric("Total apps + ARRS", f"{total_apps_arrs}")
                    with col4:
                        st.metric("Average apps per 1000 per week", f"{av_1000_week:.2f}")
                        if av_1000_week < 150:
                            st.badge("Enter Weighted list size in sidebar", color='yellow')
                        elif av_1000_week > THRESHOLD_100_PERCENT:
                            st.badge("100% Access Payment", color='green')
                        elif av_1000_week >= THRESHOLD_75_PERCENT:
                            st.badge("75% Access Payment", color='yellow')
                        else:
                            st.badge("< 75% Access Payment", color='orange')
                
                # Target Achievement Calculator
                target_metrics = calculate_target_achievement(
                    filtered_df,
                    list_size,
                    arrs_values,
                    exp_add_apps_per_week
                )
                
                if target_metrics is not None:
                    with st.expander("Target Achievement Calculator (FY 25-26)", icon=":material/calculate:", expanded=False):
                        st.subheader(":material/calculate: Target Achievement Calculator")
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Annual Target Total", f"{target_metrics['annual_target_total']:.0f}")
                            st.caption("85 apps/1000/week")
                        with c2:
                            st.metric("Baseline Projection", f"{target_metrics['total_baseline_projection']:.0f}")
                            st.caption("Historical Avg + ARRS")
                        with c3:
                            color = "normal" if target_metrics['gap'] <= 0 else "inverse"
                            st.metric(
                                "Gap to Target",
                                f"{max(0, target_metrics['gap']):.0f}",
                                delta=f"{target_metrics['gap']:.0f}",
                                delta_color=color
                            )
                            st.caption("Extra apps needed")
                        
                        if target_metrics['gap'] > 0:
                            st.warning(
                                f":material/target: To reach the annual target, you need to add "
                                f"**{target_metrics['required_extra_per_week']:.1f}** additional appointments per week "
                                f"(above our mean weekly surgery apps and ARRS apps) for the remaining "
                                f"**{target_metrics['weeks_remaining']:.1f}** weeks."
                            )
                        else:
                            st.success(
                                ":material/done_outline: Based on your current average, "
                                "you are on track to hit the annual target!"
                            )
                        
                        # Projection Chart
                        st.divider()
                        st.markdown("#### :material/bar_chart: Annual Projection (Weekly)")
                        
                        proj_df = create_projection_dataframe(
                            weekly_agg,
                            target_metrics,
                            arrs_values,
                            exp_add_apps_per_week
                        )
                        
                        fig_proj = create_projection_chart(proj_df, list_size)
                        st.plotly_chart(fig_proj, width='stretch')
                
                # Clinician Stats
                with st.expander("Clinician Stats", icon=":material/stethoscope:"):
                    st.subheader(":material/stethoscope: Clinician Stats")
                    
                    # Boxplot
                    fig_box = create_duration_boxplot(filtered_df_with_dna, selected_clinicians)
                    st.plotly_chart(fig_box, width='stretch')
                    
                    st.caption(
                        "The boxplot shows the distribution of appointment durations: "
                        "the box represents the middle 50% of durations, the line inside shows the median, "
                        "and dots indicate outliers."
                    )
                    
                    st.divider()
                    st.markdown("#### Appointment Stats by Clinician")
                    
                    # Calculate stats
                    stats_df = calculate_clinician_stats(filtered_df_with_dna, selected_clinicians)
                    
                    if stats_df is not None:
                        st.dataframe(stats_df, width='stretch', hide_index=True)
                    else:
                        st.warning("No clinician data available for the selected filters.")
                
                # Debug Information
                expander_open_debug = show_dataframe
                
                with st.expander("Debug Information", icon=":material/bug_report:", expanded=expander_open_debug):
                    if len(filtered_df) == 0:
                        st.info(
                            "### :material/info: No Data Selected\n"
                            "Please select at least one **Clinician** and **Rota Type** "
                            "to view debug information and calculations."
                        )
                    else:
                        st.markdown("### :material/bug_report: Debug Info & Data Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Original Rows:** :orange[{len(combined_df)}]")
                        with col2:
                            st.markdown(f"**Filtered Rows:** :orange[{len(filtered_df)}]")
                        with col3:
                            st.markdown(f"**Columns:** :orange[{len(combined_df.columns)}]")
                        with col4:
                            st.markdown(f"**Files Loaded:** :orange[{len(file_info)}]")
                        
                        st.markdown("---")
                        st.markdown("**File Details:**")
                        for info in file_info:
                            if info['success']:
                                st.markdown(f"- `{info['name']}`: :orange[{info['rows']}] rows")
                        
                        st.divider()
                        
                        # Configuration details
                        st.markdown("#### :material/settings: Input Configuration")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Weighted List Size:** :orange[{list_size}]")
                        with col2:
                            st.markdown(f"**ARRS Entered:** :orange[{arrs_2526}]")
                        with col3:
                            st.markdown(f"**Future ARRS:** :orange[{'Yes' if arrs_future else 'No'}]")
                        with col4:
                            st.markdown(f"**Exclude DNAs:** :orange[{'Yes' if exclude_did_not_attend else 'No'}]")
                        
                        st.divider()
                        
                        # Date range details
                        st.markdown("#### :material/calendar_check: Date Range")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Slider Start:** :orange[{date_range[0]}]")
                        with col2:
                            st.markdown(f"**Slider End:** :orange[{date_range[1]}]")
                        with col3:
                            st.markdown(f"**Data Start:** :orange[{filtered_df['appointment_date'].min().date()}]")
                        with col4:
                            st.markdown(f"**Data End:** :orange[{filtered_df['appointment_date'].max().date()}]")
                        
                        st.divider()
                        
                        # Appointment counts
                        st.markdown("#### :material/call: Appointment Counts")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Filtered Apps:** :orange[{len(filtered_df)}]")
                        with col2:
                            st.markdown(f"**ARRS Applied:** :orange[{arrs_2526}]")
                        with col3:
                            st.markdown(f"**Future ARRS Est:** :orange[{arrs_values['future_arrs_apps']}]")
                        with col4:
                            st.markdown(f"**Total + ARRS:** :orange[{total_apps_arrs}]")
                        
                        st.markdown("**Formula for Total Apps:**")
                        st.code(
                            f"total_apps_arrs = total_surgery_apps ({total_surgery_apps}) + "
                            f"arrs_2526 ({arrs_2526}) + future_arrs_apps ({arrs_values['future_arrs_apps']}) = "
                            f"{total_apps_arrs}"
                        )
                        
                        st.divider()
                        
                        # Time calculations
                        st.markdown("#### :material/schedule: Time Calculations")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Days in Range:** :orange[{time_metrics['time_diff_days']}]")
                        with col2:
                            st.markdown(f"**Weeks:** :orange[{time_metrics['weeks']:.2f}]")
                        with col3:
                            st.markdown(f"**Months:** :orange[{time_metrics['months']:.1f}]")
                        
                        st.markdown("**Formula for Weeks:**")
                        st.code(f"weeks = time_diff ({time_metrics['time_diff_days']}) / 7 = {time_metrics['weeks']:.2f}")
                        
                        st.divider()
                        
                        # Final calculation
                        st.markdown("#### :material/edit: Final Calculation")
                        st.markdown(
                            f"**Formula:** `({total_apps_arrs} ÷ {list_size}) × 1000 ÷ {time_metrics['safe_weeks']:.2f}` = "
                            f":orange[{av_1000_week:.2f}] apps per 1000 per week"
                        )
                        
                        if show_dataframe:
                            st.divider()
                            st.markdown("#### Dataframes")
                            st.write("**filtered_df**")
                            st.dataframe(filtered_df, width='stretch')
                            st.write("**weekly_agg**")
                            st.dataframe(weekly_agg, width='stretch')
                            st.write("**monthly_agg**")
                            st.dataframe(monthly_agg, width='stretch')
                            st.caption('Access ES Tracker')
                
                # Download CSV
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download Combined CSV",
                    data=csv,
                    file_name="combined_data.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning("Please select at least one clinician and rota type to display.")
        else:
            st.warning(
                f"Required columns ({', '.join(REQUIRED_COLUMNS)}) not found in the uploaded data."
            )

else:
    st.info("### :material/line_start_arrow: Please upload CSV files using the sidebar to get started!")
    st.info(
        """Create an **appointment report** in clinical reporting on SystmOne for the time period required. 
        **Breakdown** search with the following columns:
        
        - Appointment Date  
        - Appointment duration (actual)
        - Appointment Status  
        - Rota type
        - Appointment flags  
        - Clinician  
        - Appointment status 
        - Time between booking and appointment 
        - Patient ID  
        
Export as multiple searches (incrementing date with each) as SystmOne only allow an export of 30 000 rows per search.  
Download the search below and input into SystmOne.  
Any questions please email jan.duplessis@nhs.net"""
    )
