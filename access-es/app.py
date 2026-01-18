import streamlit as st
import pandas as pd
import io
from config import *
from utils import *
from plots import *
from nbeats_forecasting import (
    train_nbeats_model_with_covariates,
    forecast_nbeats_model,
    plot_validation_and_forecast,
    create_train_data,
  
)
from train_model import (
    prepare_training_data,
    train_nbeats_model,
    forecast_with_trained_model,
)

# Page Configuration
st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT, page_icon=PAGE_ICON)

# Header
st.title(f":blue[{PAGE_ICON}] {PAGE_TITLE}")
st.caption(
    "Check your **:material/switch_access_shortcut_add: Access ES** appointment provision. "
    "To ensure accurate results complete all the **settings** in the **sidebar**. "
    "Especially ARRS up to the date this value has been supplied by the ICB."
)
st.logo("images/logo.png", size="small")

# Sidebar Configuration
with st.sidebar:
    st.image("images/m_men.png")
    st.title(":material/settings: Settings")

    # File Upload
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Select one or more CSV files to combine",
    )

    # Processing Options
    drop_duplicates = st.toggle("Drop Duplicate Entries", value=False)
    exclude_did_not_attend = st.toggle(
        "Exclude 'Did Not Attend'",
        value=True,
        help="Filter out appointments with **'Did Not Attend'** status",
    )

    if exclude_did_not_attend:
        st.info(":material/done_outline: 'Did Not Attend' appointments excluded.")
    else:
        st.error(
            ":material/warning: Remove 'Did Not Attend' appointments, for accurate calculations."
        )

    st.divider()

    # Scatter Plot Options
    st.subheader("Scatter Plot Options")
    plot_view_option = st.radio(
        "Select scatter plot view",
        options=["Rota Type", "App Flags", "DNAs", "Rota/Flags"],
        index=0,
        help="Choose which columns to use for the scatter plot axes and color",
    )

    st.divider()

    # Weighted List Size
    st.subheader("Weighted List Size")
    list_size = st.number_input(
        "Weighted List Size",
        min_value=1,
        value=DEFAULT_LIST_SIZE,
        step=1,
        help="Enter your surgery's **weighted list size**, and press **Enter**.",
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
        "Set the **date of allocation** in the select box below. Enter 0 if no ARRS data.",
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
            help="Select the month corresponding to the ARRS input.",
        )

        arrs_end_date = ARRS_MONTH_MAP[arrs_month]
        arrs_start_date = FY_START
        arrs_weeks_span = max(
            0.1, (arrs_end_date - arrs_start_date).days / DAYS_PER_WEEK
        )
        estimated_weekly_arrs = arrs_2526 / arrs_weeks_span

        arrs_future = st.toggle(
            "Apply estimated future ARRS?",
            value=False,
            help="Estimate future ARRS based on current weekly rate",
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
        help="Simulate adding more appointments per week for the REMAINING period of the financial year.",
    )

    exp_add_total_apps = st.slider(
        "Experiment: Add Total Apps",
        min_value=0,
        max_value=5000,
        value=0,
        step=50,
        help="Simulate adding total appointments to the historical data. "
        "This affects the Average apps per 1000 per week calculation.",
    )

    st.divider()

    # Model Training Toggle
    show_model_training = st.toggle(
        ":material/school: Enable Model Training",
        value=False,
        help="Display the Model Training expander section for NBEats neural network training.",
    )

    st.divider()

    # Display Options
    show_dataframe = st.checkbox(
        ":material/bug_report: Debug + Export Mode",
        value=False,
        help="Display debug expander with calculations and dataframes.",
    )

    # Download Filtered Data Button
    if 'filtered_df_for_download' in st.session_state and st.session_state['filtered_df_for_download'] is not None:
        csv_download = st.session_state['filtered_df_for_download'].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv_download,
            file_name="filtered_appointments.csv",
            mime="text/csv",
            type="secondary",
            icon=":material/download:",
            help="Download the filtered dataset with all date and rota type filters applied.",
            use_container_width=True,
        )
    else:
        st.info(":material/info: Upload data and apply filters to enable download.")

# Main Content Area
if uploaded_files:
    # Load and Combine Files
    combined_df, file_info = load_and_combine_csv_files(uploaded_files)

    # Display file loading errors
    for info in file_info:
        if not info["success"]:
            st.sidebar.error(
                f":material/close: Error loading {info['name']}: {info['error']}"
            )

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
            combined_df = combined_df.drop_duplicates(keep="first")

        # Check required columns
        has_required_cols = all(col in combined_df.columns for col in REQUIRED_COLUMNS)

        if has_required_cols:
            # Get unique values for filters
            all_clinicians = sorted(combined_df["clinician"].unique().tolist())
            all_rota_types = sorted(combined_df["rota_type"].unique().tolist())

            # Create filter columns
            col1, col2 = st.columns(2)

            with col1:
                selected_rota_types = st.multiselect(
                    "Select Rota Types to Display",
                    options=all_rota_types,
                    default=all_rota_types,
                    help="Select which rota types to show",
                )

            # Default clinicians based on selected rota types
            if selected_rota_types:
                default_clinicians = sorted(
                    combined_df[combined_df["rota_type"].isin(selected_rota_types)][
                        "clinician"
                    ]
                    .unique()
                    .tolist()
                )
            else:
                default_clinicians = []

            with col2:
                selected_clinicians = st.multiselect(
                    "Select Clinicians to Display",
                    options=all_clinicians,
                    default=default_clinicians,
                    help="All clinicians are available. Defaults to clinicians in selected rota types",
                )

            if selected_clinicians and selected_rota_types:
                # Apply filters
                filtered_df, filtered_df_with_dna, dna_count, dna_percentage = (
                    filter_dataframe(
                        combined_df,
                        selected_clinicians,
                        selected_rota_types,
                        date_range=None,  # Will be set after slider
                        exclude_dna=False,  # Will apply later
                    )
                )

                # Date Range Slider
                min_date = filtered_df["appointment_date"].min().date()
                max_date = filtered_df["appointment_date"].max().date()

                if min_date == max_date:
                    date_range = (min_date, max_date)
                else:
                    date_range = st.slider(
                        "Select Date Range",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date),
                        help="Filter appointments by date range",
                    )

                # Apply date filter
                filtered_df = filtered_df[
                    (filtered_df["appointment_date"].dt.date >= date_range[0])
                    & (filtered_df["appointment_date"].dt.date <= date_range[1])
                ]
                filtered_df_with_dna = filtered_df_with_dna[
                    (filtered_df_with_dna["appointment_date"].dt.date >= date_range[0])
                    & (
                        filtered_df_with_dna["appointment_date"].dt.date
                        <= date_range[1]
                    )
                ]

                # Show duplicate handling badge
                if drop_duplicates:
                    st.badge(
                        f":material/done_outline: Dropped **{dup_count}** duplicate rows.",
                        color="blue",
                    )
                else:
                    st.badge(
                        f":material/warning: **{dup_count}** Duplicate rows identified.",
                        color="yellow",
                    )

                # Apply DNA exclusion
                dna_count_before = len(filtered_df)
                if exclude_did_not_attend:
                    filtered_df = filtered_df[
                        filtered_df["appointment_status"] != "Did Not Attend"
                    ]
                    dna_count = dna_count_before - len(filtered_df)
                    dna_percentage = (
                        (dna_count / dna_count_before * 100)
                        if dna_count_before > 0
                        else 0
                    )
                    st.badge(
                        f":material/done_outline: Excluded {dna_count} 'Did Not Attend' appointments "
                        f"({dna_percentage:.1f}% of total)",
                        color="blue",
                    )

                # Store filtered_df in session state for download button
                st.session_state['filtered_df_for_download'] = filtered_df.copy()

                # Calculate time metrics
                time_metrics = calculate_time_metrics(date_range)

                # Calculate ARRS values
                slider_end_date = date_range[1]
                filtered_end_date = filtered_df["appointment_date"].max()

                arrs_values = calculate_arrs_values(
                    arrs_2526,
                    arrs_month,
                    arrs_future,
                    slider_end_date,
                    arrs_end_date,
                    time_metrics["weeks"],
                )
                arrs_values["arrs_2526"] = arrs_2526
                arrs_values["arrs_end_date"] = arrs_end_date
                arrs_values["arrs_future"] = arrs_future

                # Calculate total appointments
                total_surgery_apps = len(filtered_df) + exp_add_total_apps

                if arrs_values["arrs_future"] and arrs_values["should_apply_arrs"]:
                    total_apps_arrs = (
                        total_surgery_apps + arrs_2526 + arrs_values["future_arrs_apps"]
                    )
                elif arrs_values["should_apply_arrs"]:
                    total_apps_arrs = total_surgery_apps + arrs_2526
                else:
                    total_apps_arrs = total_surgery_apps

                av_1000_week = calculate_apps_per_1000(
                    total_apps_arrs, list_size, time_metrics["safe_weeks"]
                )

                # Create aggregations
                cutoff_date = ARRS_MONTH_MAP[arrs_month]

                weekly_agg = create_weekly_aggregation(
                    filtered_df, list_size, arrs_values, cutoff_date
                )

                monthly_agg = create_monthly_aggregation(
                    filtered_df, list_size, arrs_values, cutoff_date
                )

                # Scatter Plot Visualization
                with st.expander(
                    "Visualizations - Appointments Scatter Plot",
                    icon=":material/scatter_plot:",
                    expanded=False,
                ):
                    st.subheader(":material/scatter_plot: Visualizations")

                    fig = create_scatter_plot(
                        filtered_df, plot_view_option, selected_clinicians
                    )
                    st.plotly_chart(fig, width="stretch")

                    # Duration statistics
                    st.badge(
                        f"Average Appointment Length: **{filtered_df['duration'].mean():.2f} minutes** "
                        f"max: {filtered_df['duration'].max():.2f} minutes ",
                        icon=":material/health_cross:",
                    )
                    st.badge(
                        f"Average Time Booking to Appointment: **{filtered_df['book_to_app'].mean():.2f} minutes** "
                        f"max: {filtered_df['book_to_app'].max():.2f} minutes ",
                        icon=":material/health_cross:",
                    )

                # Weekly & Monthly Trends
                with st.expander(
                    "Visualizations - Weekly & Monthly Trends",
                    icon=":material/bar_chart:",
                    expanded=False,
                ):
                    st.subheader(":material/bar_chart: Weekly & Monthly Trends")

                    # Monthly plot
                    fig_monthly = create_monthly_trend_plot(
                        monthly_agg, arrs_end_date, arrs_values["should_apply_arrs"]
                    )
                    st.plotly_chart(fig_monthly, width="stretch")

                    # Toggle for weekly view
                    show_per_1000 = st.toggle(
                        "Apps per 1000 per week",
                        value=False,
                        help="Toggle between Actual Appointment counts (Bar) and Apps per 1000 per week (Line)",
                    )

                    # Weekly plot
                    fig_weekly = create_weekly_trend_plot(
                        weekly_agg,
                        show_per_1000,
                        list_size,
                        arrs_end_date,
                        arrs_values["should_apply_arrs"],
                    )
                    st.plotly_chart(fig_weekly, width="stretch")

                # Data Statistics
                expander_open = not show_dataframe

                with st.expander(
                    "**Data Statistics**",
                    icon=":material/database:",
                    expanded=expander_open,
                ):
                    st.subheader(":material/database: Data Statistics")

                    col1, col2 = st.columns([1, 6])
                    with col1:
                        if drop_duplicates:
                            st.badge(
                                ":material/done_outline: Duplicates dropped",
                                color="green",
                                help=":material/info: Duplicated can be left but try with and without.",
                            )
                        else:
                            st.badge(
                                ":material/warning: Duplicate entries", color="yellow"
                            )

                    with col2:
                        if exclude_did_not_attend:
                            st.badge(
                                ":material/done_outline: Excluding 'Did Not Attend'",
                                color="green",
                            )
                        else:
                            st.badge(
                                ":material/warning: Includes 'Did Not Attend'",
                                color="yellow",
                            )

                    # First row metrics
                    col1, col2, col3, col4 = st.columns(4)
                    start_date = (
                        filtered_df["appointment_date"].min().strftime("%d %b %y")
                    )
                    end_date = (
                        filtered_df["appointment_date"].max().strftime("%d %b %y")
                    )

                    with col1:
                        st.metric("Total Surgery Appointments", total_surgery_apps)
                        st.badge(
                            f"{total_surgery_apps/list_size*1000/time_metrics['safe_weeks']:.2f} apps per 1000 per week"
                        )

                    with col2:
                        if arrs_values["should_apply_arrs"]:
                            if arrs_future:
                                total_arrs_estimated = (
                                    arrs_2526 + arrs_values["future_arrs_apps"]
                                )
                                st.metric(
                                    f"Total ARRS estimated to {end_date}",
                                    total_arrs_estimated,
                                )
                                st.badge(
                                    f"+ Future ARRS Applied! ({arrs_2526} + {arrs_values['future_arrs_apps']})",
                                    color="yellow",
                                )
                            else:
                                st.metric(f"Total ARRS to end {arrs_month}", arrs_2526)
                                st.badge(
                                    f"ARRS Applied till {arrs_month}", color="green"
                                )
                        else:
                            st.metric("Total ARRS Applied", 0)
                            st.badge(
                                f"No ARRS - End date before {arrs_month}",
                                color="orange",
                            )

                    with col3:
                        st.metric("Start Date", start_date)
                    with col4:
                        st.metric("End Date", end_date)

                    # Second row metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Time Range (Weeks)", f"{time_metrics['weeks']:.1f}")
                    with col2:
                        st.metric(
                            "Time Range (Months)", f"{time_metrics['months']:.1f}"
                        )
                    with col3:
                        st.metric("Total apps + ARRS", f"{total_apps_arrs}")
                    with col4:
                        st.metric(
                            "Average apps per 1000 per week", f"{av_1000_week:.2f}"
                        )
                        if av_1000_week < 150:
                            st.badge(
                                "Enter Weighted list size in sidebar", color="yellow"
                            )
                        elif av_1000_week > THRESHOLD_100_PERCENT:
                            st.badge("100% Access Payment", color="green")
                        elif av_1000_week >= THRESHOLD_75_PERCENT:
                            st.badge("75% Access Payment", color="yellow")
                        else:
                            st.badge("< 75% Access Payment", color="orange")

                # Target Achievement Calculator
                target_metrics = calculate_target_achievement(
                    filtered_df, list_size, arrs_values, exp_add_apps_per_week
                )

                if target_metrics is not None:
                    with st.expander(
                        "Target Achievement Calculator (FY 25-26)",
                        icon=":material/calculate:",
                        expanded=False,
                    ):
                        st.subheader(
                            ":material/calculate: Target Achievement Calculator"
                        )

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(
                                "Annual Target Total",
                                f"{target_metrics['annual_target_total']:.0f}",
                            )
                            st.caption("85 apps/1000/week")
                        with c2:
                            st.metric(
                                "Baseline Projection",
                                f"{target_metrics['total_baseline_projection']:.0f}",
                            )
                            st.caption("Historical Avg + ARRS")
                        with c3:
                            color = (
                                "normal" if target_metrics["gap"] <= 0 else "inverse"
                            )
                            st.metric(
                                "Gap to Target",
                                f"{max(0, target_metrics['gap']):.0f}",
                                delta=f"{target_metrics['gap']:.0f}",
                                delta_color=color,
                            )
                            st.caption("Extra apps needed")

                        if target_metrics["gap"] > 0:
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

                        # Projection Chart with NBEats Forecasting
                        st.divider()
                        st.markdown(
                            "#### :material/bar_chart: Annual Projection (Weekly)"
                        )

                        # Check if there's a stored test forecast from the training section
                        forecast_df = None

                        # First priority: Use the test forecast if available (from Generate Test Forecast button)
                        if "trajectory_forecast" in st.session_state:
                            forecast_df = st.session_state["trajectory_forecast"]
                            st.success(
                                f":material/online_prediction: Using **test forecast** from trained model. "
                                f"Showing {len(forecast_df)} weeks of predictions."
                            )
                        # Second priority: Check if trained model exists for live forecasting
                        elif (
                            st.session_state.get("model_trained", False)
                            and "trained_model" in st.session_state
                        ):
                            # Use trained model for forecasting
                            with st.spinner(
                                "Generating forecast with trained model..."
                            ):
                                try:
                                    forecast_df = forecast_with_trained_model(
                                        st.session_state["trained_model"],
                                        forecast_weeks=int(
                                            target_metrics["weeks_remaining"]
                                        ),
                                    )

                                    if forecast_df is not None:
                                        st.success(
                                            f"🤖 Using **trained model** ({st.session_state.get('training_weeks', 0)} weeks training data). "
                                            f"Predicted {len(forecast_df)} weeks of appointments."
                                        )
                                    else:
                                        st.warning(
                                            "Trained model forecast failed. Using baseline projection."
                                        )
                                except Exception as e:
                                    st.warning(
                                        f"Trained model error: {str(e)}. Using baseline projection."
                                    )
                                    forecast_df = None
                        else:
                            forecast_df = None

                        # Create projection dataframe with or without ML forecast
                        proj_df = create_projection_dataframe(
                            weekly_agg,
                            target_metrics,
                            arrs_values,
                            exp_add_apps_per_week,
                            forecast_df=forecast_df,
                        )

                        fig_proj = create_projection_chart(proj_df, list_size)
                        st.plotly_chart(fig_proj, width="stretch")

                        # Show forecast details if available
                        if forecast_df is not None:
                            with st.expander("View Forecast Details", expanded=False, icon=":material/expand_circle_down:"):
                                # Calculate metrics for the forecast period
                                total_forecast_apps = forecast_df[
                                    "forecasted_appointments"
                                ].sum()
                                num_forecast_weeks = len(forecast_df)
                                avg_forecast_per_week = (
                                    total_forecast_apps / num_forecast_weeks
                                    if num_forecast_weeks > 0
                                    else 0
                                )
                                avg_forecast_per_1000 = (
                                    avg_forecast_per_week / list_size
                                ) * 1000

                                # Calculate total from historic data + forecast
                                total_historic_apps = len(filtered_df)
                                total_combined_apps = (
                                    total_historic_apps + total_forecast_apps
                                )
                                total_weeks = time_metrics["weeks"] + num_forecast_weeks
                                avg_combined_per_1000 = (
                                    (total_combined_apps / list_size)
                                    * 1000
                                    / total_weeks
                                )

                                # Calculate total surgery appointments per 1000 for entire period
                                total_surgery_per_1000 = (
                                    total_combined_apps / list_size
                                ) * 1000

                                # Calculate historic apps per 1000 per week
                                historic_apps_per_1000_per_week = (
                                    (total_historic_apps / list_size) * 1000 / time_metrics["weeks"]
                                    if time_metrics["weeks"] > 0
                                    else 0
                                )

                                # Calculate forecast apps per 1000 per week
                                forecast_apps_per_1000_per_week = (
                                    (total_forecast_apps / list_size) * 1000 / num_forecast_weeks
                                    if num_forecast_weeks > 0
                                    else 0
                                )

                                # Calculate average weekly ARRS across entire period
                                # ARRS for historic period (up to arrs_month)
                                historic_weeks_with_arrs = time_metrics["weeks"]
                                historic_arrs = (
                                    arrs_values["estimated_weekly_arrs"]
                                    * historic_weeks_with_arrs
                                )

                                # ARRS for forecast period (future months)
                                forecast_arrs = (
                                    arrs_values["estimated_weekly_arrs"]
                                    * num_forecast_weeks
                                )

                                # Total ARRS and average per week
                                total_arrs = historic_arrs + forecast_arrs
                                avg_weekly_arrs = (
                                    total_arrs / total_weeks if total_weeks > 0 else 0
                                )
                                avg_weekly_arrs_per_1000 = (
                                    avg_weekly_arrs / list_size
                                ) * 1000

                                # Combined Surgery + ARRS per 1000 per week
                                combined_surgery_arrs_per_1000_per_week = (
                                    avg_combined_per_1000 + avg_weekly_arrs_per_1000
                                )

                                # Display metrics - Row 1: Surgery Appointments Breakdown
                                st.markdown("**Surgery Appointments (Historic + Forecast):**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Historic Surgery Apps",
                                        f"{total_historic_apps:,}",
                                    )
                                    st.caption(f"{time_metrics['weeks']:.1f} weeks")
                                with col2:
                                    st.metric(
                                        "Forecast Surgery Apps",
                                        f"{int(total_forecast_apps):,}",
                                    )
                                    st.caption(f"{num_forecast_weeks} weeks")
                                with col3:
                                    st.metric(
                                        "Total Surgery Apps",
                                        f"{total_combined_apps:,}",
                                    )
                                    st.caption(f"{total_weeks:.1f} weeks total")
                                with col4:
                                    st.metric(
                                        "Total Surgery Apps/1000",
                                        f"{total_surgery_per_1000:.2f}",
                                    )
                                    st.caption("Absolute total")

                                # Display metrics - Row 2: Surgery Apps per 1000 per Week
                                st.markdown("**Surgery Apps per 1000 per Week:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Historic per 1000/Week",
                                        f"{historic_apps_per_1000_per_week:.2f}",
                                    )
                                with col2:
                                    st.metric(
                                        "Forecast per 1000/Week",
                                        f"{forecast_apps_per_1000_per_week:.2f}",
                                    )
                                with col3:
                                    st.metric(
                                        "Combined Avg per 1000/Week",
                                        f"{avg_combined_per_1000:.2f}",
                                    )
                                    st.caption("Surgery only")
                                with col4:
                                    st.metric(
                                        "Forecast Avg/Week",
                                        f"{avg_forecast_per_week:.0f}",
                                    )
                                    st.caption("Absolute appointments")

                                # Display metrics - Row 3: ARRS Metrics
                                st.markdown("**ARRS Metrics (Historic + Forecast):**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Total ARRS", f"{int(total_arrs):,}"
                                    )
                                    st.caption(f"{total_weeks:.1f} weeks")
                                with col2:
                                    st.metric(
                                        "Avg Weekly ARRS", f"{avg_weekly_arrs:.0f}"
                                    )
                                with col3:
                                    st.metric(
                                        "ARRS per 1000/Week",
                                        f"{avg_weekly_arrs_per_1000:.2f}",
                                    )
                                with col4:
                                    st.metric(
                                        "Surgery + ARRS per 1000/Week",
                                        f"{combined_surgery_arrs_per_1000_per_week:.2f}",
                                    )
                                    st.caption(":material/target: Total FY rate")

                                # Calculate shortfall to 85 per 1000 per week target
                                target_rate_per_1000 = 85.0
                                shortfall_per_1000_per_week = max(0, target_rate_per_1000 - combined_surgery_arrs_per_1000_per_week)
                                
                                # Convert to real figures for list size
                                shortfall_apps_per_week = (shortfall_per_1000_per_week / 1000) * list_size
                                total_shortfall_apps = shortfall_apps_per_week * total_weeks
                                
                                # Display metrics - Row 4: Shortfall Analysis
                                st.markdown("**Shortfall to 85 per 1000/Week Target:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    color_indicator = "🟢" if shortfall_per_1000_per_week == 0 else "🔴"
                                    st.metric(
                                        "Shortfall per 1000/Week",
                                        f"{shortfall_per_1000_per_week:.2f}",
                                    )
                                    st.caption(f"{color_indicator} Gap to target")
                                with col2:
                                    st.metric(
                                        "Shortfall Apps/Week",
                                        f"{shortfall_apps_per_week:.0f}",
                                    )
                                    st.caption(f"Real figures (list size: {list_size:,})")
                                with col3:
                                    st.metric(
                                        "Total Apps Needed",
                                        f"{int(total_shortfall_apps):,}",
                                    )
                                    st.caption(f"Across {total_weeks:.1f} weeks")
                                with col4:
                                    if shortfall_per_1000_per_week > 0:
                                        st.metric(
                                            "Status",
                                            "Below Target",
                                        )
                                        st.caption("⚠️ Additional capacity needed")
                                    else:
                                        st.metric(
                                            "Status",
                                            "On Target",
                                        )
                                        st.caption("✅ Meeting 85/1000/week")

                                st.caption(
                                    f"Combined period: {time_metrics['weeks']:.1f} weeks historic + {num_forecast_weeks} weeks forecast = {total_weeks:.1f} total weeks"
                                )

                                st.divider()
                                st.dataframe(
                                    forecast_df.style.format(
                                        {"forecasted_appointments": "{:.0f}"}
                                    ),
                                    width="stretch",
                                    hide_index=True,
                                )
                st.space(size="small")

                # Clinician Stats
                with st.expander("Clinician Stats", icon=":material/stethoscope:"):
                    st.subheader(":material/stethoscope: Clinician Stats")

                    # Boxplot
                    fig_box = create_duration_boxplot(
                        filtered_df_with_dna, selected_clinicians
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    st.caption(
                        "The boxplot shows the distribution of appointment durations: "
                        "the box represents the middle 50% of durations, the line inside shows the median, "
                        "and dots indicate outliers."
                    )

                    st.divider()
                    st.markdown("#### Appointment Stats by Clinician")

                    # Calculate stats
                    stats_df = calculate_clinician_stats(
                        filtered_df_with_dna, selected_clinicians
                    )

                    if stats_df is not None:

                        fig_histograms = create_clinician_stats_histograms(stats_df)
                        st.pyplot(fig_histograms)
                        # Add 1x4 histogram subplot using seaborn
                        st.divider()
                        st.dataframe(stats_df, width="stretch", hide_index=True)
                    else:
                        st.warning(
                            "No clinician data available for the selected filters."
                        )

                # Debug Information
                expander_open_debug = show_dataframe

                # Generate and Display Training Dataset -------------------------------------------------------------------------------------------------
                if show_model_training:
                    # Auto-expand if model is trained or training dataset is ready
                    expand_training = (
                        st.session_state.get("nbeats_model_trained", False)
                        or st.session_state.get("train_ts") is not None
                    )
                    with st.expander(
                        "Model Training",
                        icon=":material/school:",
                        expanded=expand_training,
                    ):
                        st.subheader(":material/school: Model Training")
                        st.caption(
                            "Train Neural Network to predict future appointments demand."
                        )
                        st.caption(
                            "From SystmOne download historic appointment data for 2 - 3 years, breakdown to `Appointmetn date` `Appointment status` `Clinician` `Rota type` `Patient ID`, upload your csv files - influeanza data is provided centrally and goes back to 2021. Adjust the Date Slider to select where you want prediction to start. Your current dataset loaded at the top of the app is used for the current years training data. Remeber to filter your training data and exlude ARRS Staff. To do this load your training data at the top of the app first, filter as needed and check the `Debug + Export Mode` in the sidebar to export, now import this as your training data."
                        )

                        # Upload additional training data
                        training_files = st.file_uploader(
                            "Upload training dataset",
                            type="csv",
                            accept_multiple_files=True,
                            help="Prepare a dataset from **1 Jan 2021 - 30 April 2025** from the same search you used to create your initial dataset. After your run your search break down only to **Appointment date, Appointment Status, Patient ID**.",
                            key="training_upload",
                        )

                        if not training_files or len(training_files) == 0:
                            st.info(
                                "⬆️ Upload training dataset files above to begin model training."
                            )
                            st.stop()

                        st.caption("Select columns from your training data")

                        # Get column names from uploaded files or use defaults
                        available_columns = [
                            "Date",
                            "used_apps",
                        ]  # Defaults for historical data

                        try:
                            # Read first uploaded file to get column names
                            first_file = training_files[0]
                            first_file.seek(0)  # Reset file pointer
                            sample_df = pd.read_csv(first_file)

                            # Normalize column names to lowercase with underscores
                            sample_df.columns = sample_df.columns.str.lower().str.replace(" ", "_").str.strip()
                            available_columns = list(sample_df.columns)
                            first_file.seek(0)  # Reset again for actual use
                            st.success(
                                f"✓ Detected {len(available_columns)} columns from uploaded file"
                            )
                        except Exception as e:
                            st.warning(
                                f"Could not read columns from uploaded file. Using defaults."
                            )

                        # Find best default for date column
                        date_default_idx = 0
                        for idx, col in enumerate(available_columns):
                            if "date" in col.lower():
                                date_default_idx = idx
                                break

                        date_column_name = st.selectbox(
                            "Select Date column",
                            options=available_columns,
                            index=date_default_idx,
                            help="Select the column containing dates",
                        )

                        # Find best default for appointments column
                        app_default_idx = min(1, len(available_columns) - 1)
                        for idx, col in enumerate(available_columns):
                            if any(
                                keyword in col.lower()
                                for keyword in ["app", "count", "used", "total"]
                            ):
                                app_default_idx = idx
                                break

                        appointments_column_name = st.selectbox(
                            "Select Appointment Status Column",
                            options=available_columns,
                            index=app_default_idx,
                            help="Select the column containing appointment status (will filter for 'Finished')",
                        )

                        # Train button - Load training data only when clicked
                        if st.button(
                            "Load Training Data",
                            type="primary",
                            use_container_width=True,
                            icon=":material/rocket_launch:",
                            help="When making adjustments to the Date Slider to set the date prediction should start on, remember to click this button again to apply you changed date."
                        ):
                            st.session_state["load_training_data"] = True
                            # Clear any previously loaded data
                            if "train_ts" in st.session_state:
                                del st.session_state["train_ts"]
                            if "training_metadata" in st.session_state:
                                del st.session_state["training_metadata"]

                        # Display training status
                        if (
                            "model_trained" in st.session_state
                            and st.session_state["model_trained"]
                        ):
                            st.success(
                                f"✅ Model trained! {st.session_state.get('training_weeks', 0)} weeks of data used."
                            )

                        st.divider()

                        # Only process training data when button is clicked OR data already loaded
                        should_process = st.session_state.get(
                            "load_training_data", False
                        )
                        data_already_loaded = (
                            "train_ts" in st.session_state
                            and st.session_state["train_ts"] is not None
                        )

                        if should_process and not data_already_loaded:
                            try:
                                # Load training files
                                training_dfs = []
                                for idx, file in enumerate(training_files):
                                    file.seek(0)  # Reset file pointer to beginning
                                    df = pd.read_csv(file)

                                    # Normalize column names to lowercase with underscores
                                    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()

                                    training_dfs.append(df)

                                if training_dfs:
                                    combined_training = pd.concat(
                                        training_dfs, ignore_index=True
                                    )
                                    st.success(
                                        f"✓ Loaded {len(training_dfs)} training files with {len(combined_training)} total rows"
                                    )

                                    # Generate training dataset using consolidated function
                                    with st.spinner(
                                        "Preparing training dataset...", show_time=True
                                    ):
                                        # Use the new create_train_data function
                                        # filtered_df contains current year data (from slider)
                                        # combined_training contains historic data
                                        train_ts, fig = create_train_data(
                                            filtered_df,
                                            combined_training,
                                            join_date=pd.Timestamp('2025-04-01')
                                        )

                                        # Create metadata for display
                                        metadata = {
                                            'success': True,
                                            'total_weeks': len(train_ts),
                                            'start_date': train_ts.start_time(),
                                            'end_date': train_ts.end_time(),
                                            'training_weeks': len(train_ts),
                                            'current_weeks': 0,
                                            'components': train_ts.components.tolist(),
                                            'message': f'Successfully created training data with {len(train_ts)} weeks'
                                        }

                                    if metadata["success"]:
                                        # Store in session state
                                        st.session_state["train_ts"] = train_ts
                                        st.session_state["training_metadata"] = metadata
                                        st.session_state["load_training_data"] = (
                                            False  # Reset flag
                                        )
                                        st.rerun()  # Rerun to show the loaded data
                                    else:
                                        st.error(f"❌ {metadata['message']}")
                                        st.session_state["load_training_data"] = False

                            except Exception as e:
                                st.error(f"Error generating training dataset: {str(e)}")
                                st.session_state["load_training_data"] = False

                        # Display loaded training data if available
                        if data_already_loaded:
                            metadata = st.session_state.get("training_metadata", {})
                            train_ts = st.session_state["train_ts"]

                            if metadata.get("success", False):
                                # Display metadata
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Weeks", metadata["total_weeks"])
                                with col2:
                                    st.metric(
                                        "Training Weeks", metadata["training_weeks"]
                                    )
                                with col3:
                                    st.metric(
                                        "Current Weeks", metadata["current_weeks"]
                                    )
                                with col4:
                                    st.metric("Components", len(metadata["components"]))

                                st.badge(
                                    f"Date Range: {metadata['start_date']} to {metadata['end_date']}",
                                    color="blue",
                                )

                  
                                # Train Model Button
                                st.divider()

                                st.write("**Set Hyperparameters:**")
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    input_chunk_length = st.number_input(
                                        "input_chunk_length",
                                        min_value=1,
                                        max_value=156,
                                        step=1,
                                        value=52,
                                    )
                                with col2:
                                    output_chunk_length = st.number_input(
                                        "output_chunk_length",
                                        min_value=1,
                                        max_value=36,
                                        step=1,
                                        value=12,
                                    )
                                with col3:
                                    n_epochs = st.number_input(
                                        "n_epochs",
                                        min_value=10,
                                        max_value=200,
                                        step=10,
                                        value=100,
                                    )
                                with col4:
                                    batch_size = st.number_input(
                                        "batch_size",
                                        min_value=16,
                                        max_value=1024,
                                        step=8,
                                        value=32,
                                    )
                                with col5:
                                    learning_rate = st.number_input(
                                        "learning_rate",
                                        min_value=0.0005,
                                        max_value=0.1,
                                        step=0.0005,
                                        value=0.0005,
                                        format="%0.4f",
                                    )

                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    num_stacks = st.number_input(
                                        "num_stacks",
                                        min_value=10,
                                        max_value=120,
                                        step=10,
                                        value=30,
                                    )
                                with col2:
                                    num_blocks = st.number_input(
                                        "num_blocks",
                                        min_value=1,
                                        max_value=10,
                                        step=1,
                                        value=1,
                                    )
                                with col3:
                                    num_layers = st.number_input(
                                        "num_layers",
                                        min_value=1,
                                        max_value=32,
                                        step=1,
                                        value=4,
                                    )
                                with col4:
                                    layer_widths = st.number_input(
                                        "layer_widths",
                                        min_value=32,
                                        max_value=2048,
                                        step=8,
                                        value=512,
                                    )
                                with col5:
                                    validation_split = st.number_input(
                                        "validation_split",
                                        min_value=0.1,
                                        max_value=0.8,
                                        step=0.1,
                                        value=0.2,
                                        format="%0.1f",
                                    )

                                if st.button(
                                    "Train NBEats Model",
                                    type="primary",
                                    use_container_width=True,
                                    icon=":material/neurology:",
                                    key="train_nbeats_button",
                                ):
                                    # Spinner with status messages
                                    with st.spinner("Training NBEATS Model...", show_time=True):
                                        status_messages = []

                                        def status_callback(msg):
                                            status_messages.append(msg)

                                        trained_result = train_nbeats_model_with_covariates(
                                            train_ts,
                                            input_chunk_length=input_chunk_length,
                                            output_chunk_length=output_chunk_length,
                                            n_epochs=n_epochs,
                                            num_stacks=num_stacks,
                                            num_blocks=num_blocks,
                                            num_layers=num_layers,
                                            layer_widths=layer_widths,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            validation_split=validation_split,
                                            status_callback=status_callback,
                                        )

                                    # Show status after training completes
                                    if trained_result["success"]:
                                        st.status(
                                            "Training Complete!",
                                            state="complete",
                                            expanded=False
                                        )
                                        with st.expander("Training Details", expanded=False):
                                            for msg in status_messages:
                                                st.write(f"• {msg}")

                                        # Store hyperparameters used for training
                                        trained_result['hyperparameters'] = {
                                            'input_chunk_length': input_chunk_length,
                                            'output_chunk_length': output_chunk_length,
                                            'n_epochs': n_epochs,
                                            'num_stacks': num_stacks,
                                            'num_blocks': num_blocks,
                                            'num_layers': num_layers,
                                            'layer_widths': layer_widths,
                                            'batch_size': batch_size,
                                            'learning_rate': learning_rate,
                                            'validation_split': validation_split,
                                        }
                                        st.session_state["nbeats_trained_model"] = (
                                            trained_result
                                        )
                                        st.session_state["nbeats_model_trained"] = (
                                            True
                                        )
                                        st.success(
                                            f"NBEats Model Trained! {trained_result['training_weeks']} weeks used."
                                        )
                                        st.rerun()
                                    else:
                                        st.status(
                                            "Training Failed",
                                            state="error",
                                            expanded=True
                                        )
                                        with st.expander("Error Details"):
                                            st.error(trained_result["message"])

                                # Show if model already trained
                                if st.session_state.get("nbeats_model_trained", False):
                                    trained_result = st.session_state[
                                        "nbeats_trained_model"
                                    ]

                                    # Check if hyperparameters have changed
                                    stored_hp = trained_result.get('hyperparameters', {})
                                    current_hp = {
                                        'input_chunk_length': input_chunk_length,
                                        'output_chunk_length': output_chunk_length,
                                        'n_epochs': n_epochs,
                                        'num_stacks': num_stacks,
                                        'num_blocks': num_blocks,
                                        'num_layers': num_layers,
                                        'layer_widths': layer_widths,
                                        'batch_size': batch_size,
                                        'learning_rate': learning_rate,
                                        'validation_split': validation_split,
                                    }
                                    hyperparameters_changed = stored_hp != current_hp

                                    if hyperparameters_changed:
                                        st.warning(
                                            "⚠️ Hyperparameters have changed since the model was trained. "
                                            "Please click 'Train NBEats Model' to retrain with new settings."
                                        )
                                        st.info(
                                            f"Last trained with: input_chunk_length={stored_hp.get('input_chunk_length')}, "
                                            f"output_chunk_length={stored_hp.get('output_chunk_length')}, "
                                            f"num_blocks={stored_hp.get('num_blocks')}, "
                                            f"num_layers={stored_hp.get('num_layers')}, "
                                            f"layer_widths={stored_hp.get('layer_widths')}, "
                                            f"validation_split={stored_hp.get('validation_split')}"
                                        )
                                    else:
                                        st.success(
                                            "✅ NBEats model is trained and ready for forecasting!"
                                        )
                                        st.info(
                                            f"Training: {trained_result['training_weeks']} weeks | Validation: {trained_result['validation_weeks']} weeks"
                                        )

                                    # Display validation metrics
                                    if "metrics" in trained_result:
                                        st.markdown("**Model Performance Metrics:**")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.badge(
                                                f"sMAPE: {trained_result['metrics']['smape']:.2f}%",
                                                color="blue",
                                            )
                                        with col2:
                                            st.badge(
                                                f"MAPE: {trained_result['metrics']['mape']:.2f}%",
                                                color="blue",
                                            )
                                        with col3:
                                            st.badge(
                                                f"RMSE: {trained_result['metrics']['rmse']:.4f}",
                                                color="blue",
                                            )
                                        with col4:
                                            st.badge(
                                                f"MAE: {trained_result['metrics']['mae']:.4f}",
                                                color="blue",
                                            )

                                    # Display Validation and Forecast Plot (only if hyperparameters haven't changed)
                                    if not hyperparameters_changed:
                                        st.divider()
                                        st.markdown(
                                            "#### Validation & Forecast Visualization"
                                        )
                                        with st.spinner("Generating visualization...", show_time=True):
                                            # Get training metadata for plotting
                                            training_metadata = st.session_state.get(
                                                "training_metadata", {}
                                            )
                                            fig_val_forecast = plot_validation_and_forecast(
                                                trained_result,
                                                training_metadata=training_metadata,
                                                validation_weeks=trained_result['validation_weeks'],
                                                forecast_weeks=12,
                                            )
                                            if fig_val_forecast is not None:
                                                st.plotly_chart(
                                                    fig_val_forecast,
                                                    width='stretch',
                                                )
                                            else:
                                                st.warning(
                                                    "Could not generate visualization"
                                                )

                                    # Test Forecast Button - Store forecast for use in projection chart
                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        if st.button(
                                            "Generate Test Forecast (12 weeks)",
                                            use_container_width=True,
                                            type='primary',
                                            icon=":material/add:"
                                        ):
                                            with st.spinner("Generating forecast...", show_time=True):
                                                test_forecast_df = (
                                                    forecast_nbeats_model(
                                                        trained_result,
                                                        forecast_weeks=12,
                                                    )
                                                )
                                                if test_forecast_df is not None:
                                                    # Store in session state for use in projection chart
                                                    st.session_state[
                                                        "trajectory_forecast"
                                                    ] = test_forecast_df
                                                    st.success(
                                                        "✅ Forecast generated and added to trajectory chart!"
                                                    )
                                                    st.rerun()
                                                else:
                                                    st.error(
                                                        "❌ Forecast generation failed"
                                                    )
                                    with col2:
                                        if st.button(
                                            "Clear Forecast from Chart",
                                            use_container_width=True,
                                            icon=":material/delete:"
                                        ):
                                            if (
                                                "trajectory_forecast"
                                                in st.session_state
                                            ):
                                                del st.session_state[
                                                    "trajectory_forecast"
                                                ]
                                                st.success(
                                                    "✅ Forecast cleared from trajectory chart!"
                                                )
                                                st.rerun()

                                    # Show current forecast if available
                                    if "trajectory_forecast" in st.session_state:
                                        st.info(
                                            "ℹ️ **Forecast is active** and shown in the Target Achievement Calculator chart above."
                                        )
                                        with st.expander(
                                            "View Forecast Data", expanded=False
                                        ):
                                            st.dataframe(
                                                st.session_state["trajectory_forecast"],
                                                hide_index=True,
                                                width='stretch',
                                            )

                                # Convert TimeSeries to DataFrame for display
                                st.divider()

                                # Convert to DataFrame using values() and time_index
                                train_df = pd.DataFrame(
                                    {
                                        "week": train_ts.time_index,
                                        "appointments": train_ts.univariate_component(0)
                                        .values()
                                        .flatten(),
                                        "influenza": train_ts.univariate_component(1)
                                        .values()
                                        .flatten(),
                                    }
                                )

                                # Download button for full dataset
                                csv_data = train_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Full Training Dataset",
                                    data=csv_data,
                                    file_name="training_dataset.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error(f"❌ {metadata['message']}")

                st.space(size="small")

                # Debug Information & Export CSV - Always available regardless of model training toggle
                if show_dataframe:
                    with st.expander(
                        "Debug Information & Export CSV",
                        icon=":material/bug_report:",
                        expanded=expander_open_debug,
                    ):
                        if len(filtered_df) == 0:
                            st.info(
                                "### :material/info: No Data Selected\n"
                                "Please select at least one **Clinician** and **Rota Type** "
                                "to view debug information and calculations."
                            )
                        else:
                            with st.expander(
                                "Calculations",
                                icon=":material/functions:",
                                expanded=False,
                            ):

                                st.markdown(
                                    "### :material/bug_report: Debug Info & Data Statistics"
                                )

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(
                                        f"**Original Rows:** :orange[{len(combined_df)}]"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**Filtered Rows:** :orange[{len(filtered_df)}]"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Columns:** :orange[{len(combined_df.columns)}]"
                                    )
                                with col4:
                                    st.markdown(
                                        f"**Files Loaded:** :orange[{len(file_info)}]"
                                    )

                                st.markdown("---")
                                st.markdown("**File Details:**")
                                for info in file_info:
                                    if info["success"]:
                                        st.markdown(
                                            f"- `{info['name']}`: :orange[{info['rows']}] rows"
                                        )

                                st.divider()

                                # Configuration details
                                st.markdown(
                                    "#### :material/settings: Input Configuration"
                                )
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(
                                        f"**Weighted List Size:** :orange[{list_size}]"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**ARRS Entered:** :orange[{arrs_2526}]"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Future ARRS:** :orange[{'Yes' if arrs_future else 'No'}]"
                                    )
                                with col4:
                                    st.markdown(
                                        f"**Exclude DNAs:** :orange[{'Yes' if exclude_did_not_attend else 'No'}]"
                                    )

                                st.divider()

                                # Date range details
                                st.markdown("#### :material/calendar_check: Date Range")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(
                                        f"**Slider Start:** :orange[{date_range[0]}]"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**Slider End:** :orange[{date_range[1]}]"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Data Start:** :orange[{filtered_df['appointment_date'].min().date()}]"
                                    )
                                with col4:
                                    st.markdown(
                                        f"**Data End:** :orange[{filtered_df['appointment_date'].max().date()}]"
                                    )

                                st.divider()

                                # Appointment counts
                                st.markdown("#### :material/call: Appointment Counts")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(
                                        f"**Filtered Apps:** :orange[{len(filtered_df)}]"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**ARRS Applied:** :orange[{arrs_2526}]"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Future ARRS Est:** :orange[{arrs_values['future_arrs_apps']}]"
                                    )
                                with col4:
                                    st.markdown(
                                        f"**Total + ARRS:** :orange[{total_apps_arrs}]"
                                    )

                                st.markdown("**Formula for Total Apps:**")
                                st.code(
                                    f"total_apps_arrs = total_surgery_apps ({total_surgery_apps}) + "
                                    f"arrs_2526 ({arrs_2526}) + future_arrs_apps ({arrs_values['future_arrs_apps']}) = "
                                    f"{total_apps_arrs}"
                                )

                                st.divider()

                                # Time calculations
                                st.markdown(
                                    "#### :material/schedule: Time Calculations"
                                )
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(
                                        f"**Days in Range:** :orange[{time_metrics['time_diff_days']}]"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**Weeks:** :orange[{time_metrics['weeks']:.2f}]"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Months:** :orange[{time_metrics['months']:.1f}]"
                                    )

                                st.markdown("**Formula for Weeks:**")
                                st.code(
                                    f"weeks = time_diff ({time_metrics['time_diff_days']}) / 7 = {time_metrics['weeks']:.2f}"
                                )

                                st.divider()

                                # Final calculation
                                st.markdown("#### :material/edit: Final Calculation")
                                st.markdown(
                                    f"**Formula:** `({total_apps_arrs} ÷ {list_size}) × 1000 ÷ {time_metrics['safe_weeks']:.2f}` = "
                                    f":orange[{av_1000_week:.2f}] apps per 1000 per week"
                                )

                            if show_dataframe:

                                with st.expander(
                                    "Dataframes - Current FY",
                                    icon=":material/database:",
                                    expanded=False,
                                ):
                                    st.markdown("#### :material/database: Dataframes")
                                    # 🅾️ Rename colums to original names before display (Capitalized with spaces)
                                    st.write("**filtered_df**")
                                    st.dataframe(
                                        filtered_df, width="stretch", height=150
                                    )
                                    st.write("**weekly_agg**")
                                    st.dataframe(
                                        weekly_agg, width="stretch", height=150
                                    )
                                    st.write("**monthly_agg**")
                                    st.dataframe(
                                        monthly_agg, width="stretch", height=150
                                    )
                                    st.caption("Access ES Tracker")

                                with st.expander(
                                    "Training Data - TimeSeries Plots",
                                    icon=":material/coronavirus:",
                                    expanded=False,
                                ):
                                    st.markdown(
                                        "#### :material/coronavirus: Influenza TimeSeries"
                                    )
                                    inf, fig = process_influenza_data()
                                    st.pyplot(fig)
                                    show_df = st.checkbox(
                                        "Show df:", value=False, key="show-df3"
                                    )
                                    if show_df:
                                        st.dataframe(inf, width="stretch", height=150)

                                    st.markdown(
                                        "#### :material/database: Historic Appointment Data"
                                    )
                                    if (
                                        show_model_training
                                        and "training_metadata" in st.session_state
                                        and st.session_state.get(
                                            "training_metadata", {}
                                        ).get("appointments_series")
                                        is not None
                                    ):
                                        # Get the historic data from metadata
                                        training_metadata = st.session_state[
                                            "training_metadata"
                                        ]
                                        appointments_series = training_metadata.get(
                                            "appointments_series"
                                        )

                                        # Convert to DataFrame for display
                                        hist_df = pd.DataFrame(
                                            {
                                                "week": appointments_series.time_index,
                                                "appointments": appointments_series.values().flatten(),
                                            }
                                        )

                                        # Create plot
                                        import matplotlib.pyplot as plt

                                        fig_hist, ax = plt.subplots(figsize=(16, 2))
                                        ax.plot(
                                            hist_df["week"],
                                            hist_df["appointments"],
                                            color="#a33b54",
                                            linewidth=2,
                                        )
                                        ax.set_title(
                                            "Historic Training Data: Weekly Appointments",
                                            fontsize=14,
                                            fontweight="bold",
                                        )
                                        ax.set_xlabel("Week", fontsize=12)
                                        ax.set_ylabel("Appointments", fontsize=12)
                                        ax.grid(True, alpha=0.3, linewidth=0.3)
                                        plt.tight_layout()

                                        st.pyplot(fig_hist)

                                        show_df_hist = st.checkbox(
                                            "Show historic data df:",
                                            value=False,
                                            key="show-df-hist",
                                        )
                                        if show_df_hist:
                                            st.dataframe(
                                                hist_df, width="stretch", height=150
                                            )
                                    else:
                                        st.info(
                                            "Enable 'Show Model Training' in sidebar and upload training files to view this data."
                                        )

                                    st.markdown(
                                        "#### :material/merge: Merged Training Data"
                                    )
                                    if (
                                        show_model_training
                                        and "train_ts" in st.session_state
                                        and st.session_state["train_ts"] is not None
                                    ):
                                        # Get the merged TimeSeries
                                        train_ts_display = st.session_state["train_ts"]
                                        training_metadata = st.session_state.get("training_metadata", {})

                                        # Convert to DataFrame
                                        merged_display_df = pd.DataFrame(
                                            {
                                                "week": train_ts_display.time_index,
                                                "appointments": train_ts_display.univariate_component(
                                                    0
                                                )
                                                .values()
                                                .flatten(),
                                                "influenza": train_ts_display.univariate_component(
                                                    1
                                                )
                                                .values()
                                                .flatten(),
                                            }
                                        )
                                        
                                        # Display Training Dataset Date Range
                                        if training_metadata:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(
                                                    "Training Start Date",
                                                    pd.Timestamp(training_metadata['start_date']).strftime("%d %b %Y")
                                                )
                                            with col2:
                                                st.metric(
                                                    "Training End Date",
                                                    pd.Timestamp(training_metadata['end_date']).strftime("%d %b %Y")
                                                )
                                            with col3:
                                                st.metric(
                                                    "Total Weeks",
                                                    training_metadata['total_weeks']
                                                )

                                        # Create dual-axis plot
                                        fig_merged, ax1 = plt.subplots(figsize=(16, 3))

                                        color1 = "#ab271f"
                                        ax1.set_xlabel("Week", fontsize=12)
                                        ax1.set_ylabel(
                                            "Appointments", color=color1, fontsize=12
                                        )
                                        ax1.plot(
                                            merged_display_df["week"],
                                            merged_display_df["appointments"],
                                            color=color1,
                                            linewidth=2,
                                            label="Appointments",
                                        )
                                        ax1.tick_params(axis="y", labelcolor=color1)
                                        ax1.grid(True, alpha=0.3, linewidth=0.3)

                                        ax2 = ax1.twinx()
                                        color2 = "#545e6a"
                                        ax2.set_ylabel(
                                            "Influenza Level", color=color2, fontsize=12
                                        )
                                        ax2.plot(
                                            merged_display_df["week"],
                                            merged_display_df["influenza"],
                                            color=color2,
                                            linewidth=2,
                                            linestyle="-",
                                            label="Influenza",
                                        )
                                        ax2.tick_params(axis="y", labelcolor=color2)

                                        plt.title(
                                            f"Merged Training Data: Appointments & Influenza\n{len(merged_display_df)} weeks",
                                            fontsize=14,
                                            fontweight="bold",
                                            pad=20,
                                        )

                                        lines1, labels1 = (
                                            ax1.get_legend_handles_labels()
                                        )
                                        lines2, labels2 = (
                                            ax2.get_legend_handles_labels()
                                        )
                                        ax1.legend(
                                            lines1 + lines2,
                                            labels1 + labels2,
                                            loc="upper left",
                                            fontsize=10,
                                        )
                                        ax1.grid(True, alpha=0.3, linewidth=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig_merged)

                                        show_df_merged = st.checkbox(
                                            "Show merged data df:",
                                            value=False,
                                            key="show-df-merged",
                                        )
                                        if show_df_merged:
                                            st.dataframe(
                                                merged_display_df,
                                                width="stretch",
                                                height=150,
                                            )
                                    else:
                                        st.info(
                                            "Enable 'Show Model Training' in sidebar and upload training files to view merged training data."
                                        )

                if show_dataframe:
                    # 🅾️ Rename colums to original names before export
                    csv = filtered_df.to_csv(index=False)
                    st.sidebar.download_button(
                        label="Download Filtered Data",
                        data=csv,
                        file_name="access_tracker_filtered_data.csv",
                        mime="text/csv",
                        type="secondary",
                        icon=":material/download:",
                        help="Use thie button to download your aggregated, filtered csv, good for filtering your training dataset before traiining.",
                    )

            else:
                st.warning(
                    "Please select at least one clinician and rota type to display."
                )
        else:
            st.warning(
                f"Required columns ({', '.join(REQUIRED_COLUMNS)}) not found in the uploaded data."
            )

else:
    st.info(
        "### :material/line_start_arrow: Please upload CSV files using the sidebar to get started!"
    )
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
