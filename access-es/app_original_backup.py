import streamlit as st
import pandas as pd
import plotly.express as px
import io
from datetime import date
import numpy as np

st.set_page_config(
    page_title="Access ES - Appointment Tracker",
    layout="wide",
    page_icon=":material/switch_access_shortcut_add:",
)


st.title(":material/switch_access_shortcut_add: Access ES - Appointment Tracker")
st.caption(
    "Check your **:material/switch_access_shortcut_add: Access ES** appointment provision. To ensure accurate results complete all the **settings** in the **sidebar**. Especiaqlly ARRS up to the date this value has been supplied by the ICB."
)
st.logo("logo.png", size="small")
# Sidebar for file uploads
with st.sidebar:
    st.title(":material/settings: Settings")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Select one or more CSV files to combine",
    )
    drop_duplicates = st.toggle("Drop Duplicate Entries", value=True)
    exclude_did_not_attend = st.toggle(
        "Exclude 'Did Not Attend'",
        value=True,
        help="Filter out appointments with **'Did Not Attend'** status",
    )
    if exclude_did_not_attend:
        st.info(":material/done_outline: 'Did Not Attend' appointments excluded.")
    else:
        st.error("⚠️ Remove 'Did Not Attend' appointments, for accurate calculations.")
    st.divider()
    st.subheader("Scatter Plot Options")
    plot_view_option = st.radio(
        "Select scatter plot view",
        options=["Rota Type", "App Flags", "DNAs", "Rota/Flags"],
        index=0,
        help="Choose which columns to use for the scatter plot axes and color",
    )

    st.divider()

    st.subheader("Weighted List Size")
    list_size = st.number_input(
        "Weighted List Size",
        min_value=1,
        value=1,
        step=1,
        help="Enter your surgery's **weighted list size**, and press **Enter**.",
    )
    st.divider()

    st.header("ARRS Allocation")
    arrs_2526 = st.number_input(
        "Total ARRS to date.",
        min_value=0,
        value=0,
        step=1,
        help="Enter your **ARRS allocation** for the year to date and press **Enter**. Set the **date of allocation** in the select box below. Enter 0 if no ARRS data.",
    )

    # Initialize ARRS variables
    estimated_weekly_arrs = 0.0

    # Map month selections to end of month dates
    arrs_month_map = {
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
        "March 26": pd.Timestamp(2026, 3, 31),
    }

    if arrs_2526 == 0:
        arrs_future = False
        arrs_month = "April 25"
        arrs_end_date = arrs_month_map[arrs_month]
    else:
        st.error(
            "⚠️ **To prevent double counting:** If entering ARRS data here, ensure ARRS clinicians are **deselected** in the **Rota type** & **Clinician** filters above."
        )
        arrs_month = st.selectbox(
            "Select month for ARRS input",
            options=[
                "April 25",
                "May 25",
                "June 25",
                "July 25",
                "August 25",
                "September 25",
                "October 25",
                "November 25",
                "December 25",
                "January 26",
                "February 26",
                "March 26",
            ],
            index=0,
            help="Select the month corresponding to the ARRS input.",
        )

        arrs_end_date = arrs_month_map[arrs_month]
        arrs_start_date = pd.Timestamp(2025, 4, 1)
        arrs_weeks_span = max(0.1, (arrs_end_date - arrs_start_date).days / 7)

        # Always calculate estimated weekly ARRS if we have data
        estimated_weekly_arrs = arrs_2526 / arrs_weeks_span

        arrs_future = st.toggle(
            "Apply estimated future ARRS?",
            value=False,
            help="Estimate future ARRS based on current weekly rate",
        )
        if arrs_future:
            st.info(
                f":material/done_outline: Future ARRS estimation at **{estimated_weekly_arrs:.0f}** ARRS apps per week - or **{estimated_weekly_arrs/list_size*1000:.2f}** ARRS apps per 1000 per week"
            )
        else:
            st.error(
                f"⚠️ ARRS **will not be applied** to future months! Your Average Apps per 1000 per week will be underestimated."
            )

    st.divider()
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
        help="Simulate adding total appointments to the historical data. This affects the Average apps per 1000 per week calculation.",
    )

    st.divider()

    show_dataframe = st.checkbox(
        ":material/table: Show DataFrame",
        value=False,
        help="Display the filtered dataframe below the scatter plot",
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
        combined_df.columns = combined_df.columns.str.lower().str.replace(" ", "_")
        combined_df.columns = combined_df.columns.str.strip()
        combined_df.rename(
            columns={
                "appointment_duration_(actual)": "duration",
                "time_between_booking_and_appointment": "book_to_app",
            },
            inplace=True,
        )
        # Convert appointment_date column to datetime if it exists
        if "appointment_date" in combined_df.columns:
            try:
                # Try the expected format first
                combined_df["appointment_date"] = pd.to_datetime(
                    combined_df["appointment_date"], format="%d-%b-%y"
                )
            except ValueError:
                # If that fails, try auto-detection with dayfirst=True for DD-Mon-YY format
                combined_df["appointment_date"] = pd.to_datetime(
                    combined_df["appointment_date"], format="mixed", dayfirst=True
                )
            # Drop rows with invalid dates to prevent issues
            original_rows = len(combined_df)
            combined_df = combined_df.dropna(subset=["appointment_date"])
            if len(combined_df) < original_rows:
                st.sidebar.warning(
                    f"Dropped {original_rows - len(combined_df)} rows with invalid dates"
                )

            # Sort by appointment date
            combined_df = combined_df.sort_values("appointment_date").reset_index(
                drop=True
            )

            # 1. Use regex to extract numeric values for hours and minutes
            # This pattern looks for digits followed by 'h' and digits followed by 'm'
            extracted = combined_df["duration"].str.extract(
                r"(?:(?P<hours>\d+)h)?\s?(?:(?P<minutes>\d+)m)?"
            )
            # 2. Convert the extracted strings to numeric (filling missing values with 0)
            hours = pd.to_numeric(extracted["hours"]).fillna(0)
            minutes = pd.to_numeric(extracted["minutes"]).fillna(0)
            # 3. Calculate total minutes and assign to a new column
            combined_df["duration"] = (hours * 60) + minutes

            extracted = combined_df["book_to_app"].str.extract(
                r"(?:(?P<hours>\d+)h)?\s?(?:(?P<minutes>\d+)m)?"
            )
            # 2. Convert the extracted strings to numeric (filling missing values with 0)
            hours = pd.to_numeric(extracted["hours"]).fillna(0)
            minutes = pd.to_numeric(extracted["minutes"]).fillna(0)
            # 3. Calculate total minutes and assign to a new column
            combined_df["book_to_app"] = (hours * 60) + minutes

        # Scatter plot section
        st.subheader("Filters and Visualization")
        if drop_duplicates:
            st.badge(
                f":material/done_outline:  Dropped **{combined_df.duplicated().sum()}** duplicate rows.",
                color="blue",
            )
            combined_df = combined_df.drop_duplicates(keep="first")
        else:
            st.badge(
                f"⚠️ **{combined_df.duplicated().sum()}** Duplicate rows identified.",
                color="blue",
            )

        # Check if required columns exist
        required_cols = [
            "appointment_date",
            "clinician",
            "appointment_status",
            "rota_type",
        ]
        has_required_cols = all(col in combined_df.columns for col in required_cols)

        if has_required_cols:
            # Get unique clinician and rota_type names
            all_clinicians = sorted(combined_df["clinician"].unique().tolist())
            all_rota_types = sorted(combined_df["rota_type"].unique().tolist())

            # Create columns for multiselects
            col1, col2 = st.columns(2)

            with col1:
                # Multiselect for rota_type (primary filter)
                selected_rota_types = st.multiselect(
                    "Select Rota Types to Display",
                    options=all_rota_types,
                    default=all_rota_types,
                    help="Select which rota types to show",
                )

            # Get clinicians that work in the selected rota types (for default selection)
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
                # Multiselect for clinicians (all available, but default to those in selected rota types)
                selected_clinicians = st.multiselect(
                    "Select Clinicians to Display",
                    options=all_clinicians,
                    default=default_clinicians,
                    help="All clinicians are available. Defaults to clinicians in selected rota types",
                )

            if selected_clinicians and selected_rota_types:
                # Filter dataframe by selected clinicians and rota_types
                filtered_df = combined_df[
                    (combined_df["clinician"].isin(selected_clinicians))
                    & (combined_df["rota_type"].isin(selected_rota_types))
                ]

                # Date range slider
                min_date = filtered_df["appointment_date"].min().date()
                max_date = filtered_df["appointment_date"].max().date()

                # Handle case where min_date == max_date
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

                # Filter dataframe by date range
                filtered_df = filtered_df[
                    (filtered_df["appointment_date"].dt.date >= date_range[0])
                    & (filtered_df["appointment_date"].dt.date <= date_range[1])
                ]

                # Store filtered_df BEFORE DNA exclusion for clinician stats
                filtered_df_with_dna = filtered_df.copy()

                # Apply exclude_did_not_attend filter BEFORE creating weekly and monthly aggregations
                dna_count_before = len(filtered_df)
                if exclude_did_not_attend:
                    filtered_df = filtered_df[
                        filtered_df["appointment_status"] != "Did Not Attend"
                    ]
                    dna_count_excluded = dna_count_before - len(filtered_df)
                    dna_percentage = (dna_count_excluded / dna_count_before) * 100
                    st.badge(
                        f":material/done_outline: Excluded {dna_count_excluded} 'Did Not Attend' appointments ({dna_percentage:.1f}% of total)",
                        color="green",
                    )

                # Determine if ARRS should be applied based on slider end date (not data end date)
                # This ensures ARRS is applied if the user selects a date range that extends to the ARRS month end
                slider_end_raw = date_range[1]
                if isinstance(slider_end_raw, int):
                    slider_end_date = pd.Timestamp.fromordinal(slider_end_raw)
                else:
                    slider_end_date = pd.Timestamp(
                        slider_end_raw.year, slider_end_raw.month, slider_end_raw.day
                    )
                should_apply_arrs = slider_end_date >= arrs_end_date and arrs_2526 > 0

                # Store the actual data end date for time range calculations
                filtered_end_date = filtered_df["appointment_date"].max()

                # Calculate time_diff based on SLIDER range, not data range
                # This ensures the denominator (weeks) correctly reflects the period selected by the user,
                # even if there are gaps in the data (e.g. no appointments on weekends or holidays).
                d_start = pd.Timestamp(date_range[0])
                d_end = pd.Timestamp(date_range[1])

                time_diff = (d_end - d_start).days
                weeks = time_diff / 7
                months = time_diff / 30.44

                if arrs_future and should_apply_arrs:
                    # Calculate weeks from ARRS end date to slider end date for future estimation
                    if isinstance(slider_end_date, pd.Timestamp):
                        s_end = slider_end_date
                    else:
                        s_end = pd.Timestamp(slider_end_date)

                    if isinstance(arrs_end_date, pd.Timestamp):
                        a_end = arrs_end_date
                    else:
                        a_end = pd.Timestamp(arrs_end_date)

                    days_fut = (s_end - a_end).days
                    future_weeks = days_fut / 7
                    future_arrs_apps = (
                        int(round(estimated_weekly_arrs * future_weeks, 0))
                        if future_weeks > 0
                        else 0
                    )
                    total_apps_arrs = len(filtered_df) + arrs_2526 + future_arrs_apps
                elif should_apply_arrs:
                    total_apps_arrs = len(filtered_df) + arrs_2526
                    future_arrs_apps = 0
                else:
                    total_apps_arrs = len(filtered_df)
                    future_arrs_apps = 0

                # Total Surgery Appointments (Historical ONLY)
                total_surgery_apps = len(filtered_df) + exp_add_total_apps

                # Recalculate total_apps_arrs with experimental addition
                if arrs_future and should_apply_arrs:
                    total_apps_arrs = total_surgery_apps + arrs_2526 + future_arrs_apps
                elif should_apply_arrs:
                    total_apps_arrs = total_surgery_apps + arrs_2526
                else:
                    total_apps_arrs = total_surgery_apps

                # Prevent ZeroDivisionError
                safe_weeks = max(0.1, weeks)
                av_1000_week = (total_apps_arrs / list_size) * 1000 / safe_weeks
                # Create weekly aggregation for all appointments in filtered dataframe
                cutoff_date = arrs_month_map[arrs_month]

                weekly_df = filtered_df.copy()
                weekly_df["week"] = (
                    weekly_df["appointment_date"].dt.to_period("W").dt.start_time
                )
                weekly_agg = (
                    weekly_df.groupby("week")
                    .size()
                    .reset_index(name="total_appointments")
                )

                weekly_agg["per_1000"] = (
                    weekly_agg["total_appointments"] / list_size * 1000
                )
                # ARRS applied as flat rate across all weeks
                weeks_after_cutoff = weekly_agg[
                    weekly_agg["week"] >= cutoff_date
                ].shape[0]

                if arrs_future and weeks_after_cutoff > 0:
                    # Distribute the predicted value across the remaining weeks
                    post_cutoff_increment = future_arrs_apps / weeks_after_cutoff
                else:
                    post_cutoff_increment = 0

                # Apply ARRS: estimated_weekly_arrs before cutoff, post_cutoff_increment after cutoff
                # Split into two columns for different colors in the chart
                weekly_agg["arrs_historical"] = np.where(
                    weekly_agg["week"] < cutoff_date, estimated_weekly_arrs, 0
                )
                weekly_agg["arrs_future"] = np.where(
                    weekly_agg["week"] >= cutoff_date, post_cutoff_increment, 0
                )
                weekly_agg["arrs_only"] = (
                    weekly_agg["arrs_historical"] + weekly_agg["arrs_future"]
                )
                weekly_agg["total_with_arrs"] = (
                    weekly_agg["total_appointments"] + weekly_agg["arrs_only"]
                )
                weekly_agg["per_1000_with_arrs"] = weekly_agg["per_1000"] + (
                    weekly_agg["arrs_only"] / list_size * 1000
                )

                # Create monthly aggregation for all appointments in filtered dataframe
                monthly_df = filtered_df.copy()
                monthly_df["month"] = (
                    monthly_df["appointment_date"].dt.to_period("M").dt.start_time
                )
                monthly_agg = (
                    monthly_df.groupby("month")
                    .size()
                    .reset_index(name="total_appointments")
                )

                # Calculate months after cutoff for future ARRS distribution
                months_after_cutoff = monthly_agg[
                    monthly_agg["month"] >= cutoff_date
                ].shape[0]

                if arrs_future and months_after_cutoff > 0:
                    # Distribute future ARRS across remaining months
                    monthly_post_cutoff_increment = (
                        future_arrs_apps / months_after_cutoff
                    )
                else:
                    monthly_post_cutoff_increment = 0

                # Add monthly ARRS estimate (approximately 4.345 weeks per month before cutoff, distributed future ARRS after)
                # Split into two columns for different colors in the chart
                monthly_agg["arrs_historical"] = np.where(
                    monthly_agg["month"] < cutoff_date,
                    (estimated_weekly_arrs * 4.345),
                    0,
                )
                monthly_agg["arrs_future"] = np.where(
                    monthly_agg["month"] >= cutoff_date,
                    monthly_post_cutoff_increment,
                    0,
                )
                monthly_agg["total_arrs_estimated"] = (
                    monthly_agg["arrs_historical"] + monthly_agg["arrs_future"]
                )
                monthly_agg["total_with_arrs"] = (
                    monthly_agg["total_appointments"]
                    + monthly_agg["total_arrs_estimated"]
                )

                # Calculate dynamic height based on number of clinicians
                base_height = 300
                height_per_clinician = 15
                fig_height = base_height + (
                    len(selected_clinicians) * height_per_clinician
                )

                # Display plot based on selected view option
                with st.expander(
                    "Visualizations - Scatter Plots Appointments",
                    icon=":material/scatter_plot:",
                    expanded=False,
                ):
                    st.subheader(":material/scatter_plot: Visualizations")
                    if plot_view_option == "Rota Type":
                        # Y: Clinician, Hue: Rota Type
                        fig = px.strip(
                            filtered_df,
                            x="appointment_date",
                            y="clinician",
                            color="rota_type",
                            title="Appointments by Date and Clinician (Colored by Rota Type)",
                            labels={
                                "appointment_date": "Appointment Date",
                                "clinician": "Clinician",
                                "rota_type": "Rota Type",
                            },
                            height=fig_height,
                        )

                    elif plot_view_option == "App Flags":
                        # Y: Clinician, Hue: Appointment Flags
                        fig = px.strip(
                            filtered_df,
                            x="appointment_date",
                            y="clinician",
                            color="appointment_flags",
                            title="Appointments by Date and Clinician (Colored by Appointment Flags)",
                            labels={
                                "appointment_date": "Appointment Date",
                                "clinician": "Clinician",
                                "appointment_flags": "Appointment Flags",
                            },
                            height=fig_height,
                        )

                    elif plot_view_option == "DNAs":
                        # Y: Clinician, Hue: DNAs (appointment_status)
                        # Sort so 'Finished' appointments are plotted first, 'Did Not Attend' on top
                        plot_df_dna = filtered_df.copy()
                        status_order = {"Finished": 0, "Did Not Attend": 1}
                        plot_df_dna["_status_sort"] = plot_df_dna[
                            "appointment_status"
                        ].map(status_order)
                        plot_df_dna = plot_df_dna.sort_values("_status_sort")

                        fig = px.strip(
                            plot_df_dna,
                            x="appointment_date",
                            y="clinician",
                            color="appointment_status",
                            title="Appointments by Date and Clinician (Colored by DNAs)",
                            labels={
                                "appointment_date": "Appointment Date",
                                "clinician": "Clinician",
                                "appointment_status": "DNAs",
                            },
                            height=fig_height,
                        )
                        # Set marker opacity for transparency
                        fig.update_traces(marker=dict(opacity=0.6))

                    elif plot_view_option == "Rota/Flags":
                        # Y: Appointment Flags, Hue: Rota Type
                        num_flags = filtered_df["appointment_flags"].nunique()
                        flags_base_height = 300
                        height_per_flag = 15
                        flags_fig_height = flags_base_height + (
                            num_flags * height_per_flag
                        )

                        fig = px.strip(
                            filtered_df,
                            x="appointment_date",
                            y="appointment_flags",
                            color="rota_type",
                            title="Appointments by Date and Flags (Colored by Rota Type)",
                            labels={
                                "appointment_date": "Appointment Date",
                                "appointment_flags": "Appointment Flags",
                                "rota_type": "Rota Type",
                            },
                            height=flags_fig_height,
                        )
                        fig_height = flags_fig_height

                    fig.update_layout(
                        xaxis_title="Appointment Date",
                        yaxis_title=(
                            "Clinician"
                            if plot_view_option != "Rota/Flags"
                            else "Appointment Flags"
                        ),
                        hovermode="closest",
                        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                    )

                    st.plotly_chart(fig, width="stretch")
                    st.info(
                        f"Avarage Appointment length: **{filtered_df['duration'].mean():.2f} minutes** (Across all clinics selected above) Max: {filtered_df['duration'].max():.2f} minutes Min: {filtered_df['duration'].min():.2f} minutes"
                    )
                    st.info(
                        f"Average Time from Booking till Appointment: **{filtered_df['book_to_app'].mean():.2f} minutes** (Across all clinics selected above) Max: {filtered_df['book_to_app'].max():.2f} minutes Min: {filtered_df['book_to_app'].min():.2f} minutes"
                    )

                # Weekly & Monthly Trends
                with st.expander(
                    "Visualizations - Weekly & Monthly Trends",
                    icon=":material/bar_chart:",
                    expanded=False,
                ):
                    st.subheader(":material/bar_chart: Weekly & Monthly Trends")

                    # Calculate time range for calculations
                    time_diff_days = (
                        pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])
                    ).days
                    months_calc = time_diff_days / 30.4

                    # Toggle for weekly view options
                    show_per_1000 = st.toggle(
                        "Apps per 1000 per week",
                        value=False,
                        help="Toggle between Actual Appointment counts (Bar) and Apps per 1000 per week (Line)",
                    )

                    # Calculate threshold line
                    threshold = (85 / 1000) * list_size

                    # Prepare data for plotting
                    weekly_agg_plot = weekly_agg.copy()

                    if show_per_1000:
                        y_plot = "per_1000"
                        y_label = "Apps per 1000 per week"
                        threshold_plot = 85.0
                        plot_select_title = "Appointments per 1000 pts per Week"
                    else:
                        y_plot = "total_appointments"
                        y_label = "Count"
                        threshold_plot = threshold
                        plot_select_title = "Total Appointments per Week"

                    if not show_per_1000:
                        # Bar chart for total appointments with split ARRS colors
                        fig_weekly = px.bar(
                            weekly_agg_plot,
                            x="week",
                            y=[y_plot, "arrs_historical", "arrs_future"],
                            title=plot_select_title,
                            labels={
                                "week": "Week Starting Date",
                                "value": "Count",
                                "variable": "Type",
                            },
                            height=500,
                            color_discrete_map={
                                y_plot: "#0077b6",
                                "arrs_historical": "#f48c06",
                                "arrs_future": "#cb3f4e",
                            },
                        )

                        # Add custom text labels
                        # Blue segment: Actual value
                        # Orange/lighter orange segments: Show total on top segment
                        fig_weekly.update_traces(
                            selector=dict(name=y_plot),
                            text=weekly_agg_plot[y_plot],
                            textposition="inside",
                        )
                        fig_weekly.update_traces(
                            selector=dict(name="arrs_historical"),
                            text=weekly_agg_plot["arrs_historical"].round(0),
                            textposition="inside",
                        )
                        fig_weekly.update_traces(
                            selector=dict(name="arrs_future"),
                            text=weekly_agg_plot["total_with_arrs"].round(0),
                            textposition="inside",
                        )

                        # Update legend names
                        newnames = {
                            y_plot: "Actual Appointments",
                            "arrs_historical": "ARRS (Historical)",
                            "arrs_future": "ARRS (Future Est)",
                        }
                        fig_weekly.for_each_trace(
                            lambda t: t.update(name=newnames.get(t.name, t.name))
                        )

                        # Add threshold line to weekly bar plot
                        fig_weekly.add_hline(
                            y=threshold,
                            line_dash="dot",
                            line_color="#c1121f",
                            line_width=2,
                            annotation_text=f"Threshold ({threshold:.2f})",
                            annotation_position="top right",
                        )
                    else:
                        fig_weekly = px.line(
                            weekly_agg_plot,
                            x="week",
                            y=y_plot,
                            title=plot_select_title,
                            labels={"week": "Week Starting Date", y_plot: y_label},
                            height=500,
                            markers=False,
                            color_discrete_sequence=["#0077b6"],
                        )

                        # Update first line name for legend and add data labels
                        fig_weekly.update_traces(
                            name="Actual Apps per 1000",
                            showlegend=True,
                            text=weekly_agg_plot[y_plot].round(2),
                            textposition="top center",
                            mode="lines+text",
                            textfont=dict(size=10),
                        )

                        # Add second line for per_1000_with_arrs
                        fig_weekly.add_scatter(
                            x=weekly_agg_plot["week"],
                            y=weekly_agg_plot["per_1000_with_arrs"],
                            mode="lines+text",
                            name="Apps per 1000 + ARRS",
                            line=dict(dash="solid", color="#f48c06"),
                            text=weekly_agg_plot["per_1000_with_arrs"].round(2),
                            textposition="top center",
                            textfont=dict(size=9),
                        )

                        # Add threshold line to weekly plot
                        fig_weekly.add_hline(
                            y=threshold_plot,
                            line_dash="dash",
                            line_color="#c1121f",
                            line_width=2,
                            annotation_text=f"Threshold ({threshold_plot:.2f})",
                            annotation_position="top right",
                        )

                        # Add mean line for per_1000_with_arrs if in per 1000 view
                        if show_per_1000:
                            mean_val = weekly_agg_plot["per_1000_with_arrs"].mean()
                            fig_weekly.add_hline(
                                y=mean_val,
                                line_dash="dot",
                                line_color="#6a994e",
                                line_width=2,
                                annotation_text=f"Mean Apps + ARRS ({mean_val:.2f})",
                                annotation_position="top right",
                            )

                    # Add ARRS end date vertical line if applicable
                    if should_apply_arrs:
                        fig_weekly.add_vline(
                            x=pd.Timestamp(arrs_end_date),
                            line_dash="dashdot",
                            line_color="#749857",
                            line_width=2,
                        )
                        fig_weekly.add_annotation(
                            x=pd.Timestamp(arrs_end_date),
                            text="ARRS Prediction Start",
                            showarrow=False,
                            xanchor="right",
                            yanchor="top",
                        )

                    fig_weekly.update_layout(
                        xaxis_title="Week Starting Date",
                        yaxis_title="Total Appointments",
                        hovermode="x unified",
                        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                    )

                    st.plotly_chart(fig_weekly, width="stretch")

                    # Calculate average monthly appointments
                    avg_monthly_appointments = monthly_agg["total_appointments"].mean()

                    # Prepare data for monthly stacked bar chart (using pre-calculated columns)
                    monthly_agg_plot = monthly_agg.copy()

                    # Create stacked bar plot for monthly appointments with split ARRS colors
                    fig_monthly = px.bar(
                        monthly_agg_plot,
                        x="month",
                        y=["total_appointments", "arrs_historical", "arrs_future"],
                        title="Total Appointments per Month",
                        labels={"month": "Month", "value": "Count", "variable": "Type"},
                        height=400,
                        color_discrete_map={
                            "total_appointments": "#0077b6",
                            "arrs_historical": "#f48c06",
                            "arrs_future": "#cb3f4e",
                        },
                    )

                    # Add custom text labels
                    fig_monthly.update_traces(
                        selector=dict(name="total_appointments"),
                        text=monthly_agg_plot["total_appointments"],
                        textposition="inside",
                    )
                    fig_monthly.update_traces(
                        selector=dict(name="arrs_historical"),
                        text=monthly_agg_plot["arrs_historical"].round(0),
                        textposition="inside",
                    )
                    fig_monthly.update_traces(
                        selector=dict(name="arrs_future"),
                        text=monthly_agg_plot["total_with_arrs"].round(0),
                        textposition="inside",
                    )

                    # Update legend names
                    newnames_monthly = {
                        "total_appointments": "Actual Appointments",
                        "arrs_historical": "ARRS (Historical)",
                        "arrs_future": "ARRS (Future Est)",
                    }
                    fig_monthly.for_each_trace(
                        lambda t: t.update(name=newnames_monthly.get(t.name, t.name))
                    )

                    # Add average line to monthly plot
                    fig_monthly.add_hline(
                        y=avg_monthly_appointments,
                        line_dash="dot",
                        line_color="#e4af6c",
                        line_width=2,
                        annotation_text=f"Monthly Average Completed Apps ({avg_monthly_appointments:.1f})",
                        annotation_position="top left",
                    )

                    # Add ARRS end date vertical line if applicable
                    if should_apply_arrs:
                        fig_monthly.add_vline(
                            x=pd.Timestamp(arrs_end_date),
                            line_dash="dashdot",
                            line_color="#749857",
                            line_width=2,
                        )
                        fig_monthly.add_annotation(
                            x=pd.Timestamp(arrs_end_date),
                            text="ARRS Prediction Start",
                            showarrow=False,
                            xanchor="right",
                            yanchor="top",
                        )

                    fig_monthly.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Total Appointments",
                        hovermode="x unified",
                        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                    )

                    st.plotly_chart(fig_monthly, width="stretch")

                # Display filtered dataframe if checkbox is enabled
                if show_dataframe:
                    expander_open = False
                else:
                    expander_open = True

                # Display dynamic statistics based on filtered data
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
                            )

                        else:
                            st.badge("⚠️ Duplicate entries", color="yellow")

                    with col2:
                        if exclude_did_not_attend:
                            st.badge(
                                ":material/done_outline: Excluding 'Did Not Attend'",
                                color="green",
                            )
                        else:
                            st.badge("⚠️ Includes 'Did Not Attend'", color="yellow")

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
                            f"{total_surgery_apps/list_size*1000/safe_weeks:.2f} apps per 1000 per week"
                        )

                    with col2:
                        if should_apply_arrs:
                            if arrs_future:
                                total_arrs_estimated = arrs_2526 + future_arrs_apps
                                st.metric(
                                    f"Total ARRS estimated to {end_date}",
                                    total_arrs_estimated,
                                )
                                st.badge(
                                    f"+ Future ARRS Applied! ({arrs_2526} + {future_arrs_apps})",
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

                    # Second row of metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Time Range (Weeks)", f"{weeks:.1f}")
                    with col2:
                        st.metric("Time Range (Months)", f"{months:.1f}")
                    with col3:
                        st.metric("Total apps + ARRS", f"{total_apps_arrs}")
                    with col4:
                        st.metric(
                            "Average apps per 1000 per week", f"{av_1000_week:.2f}"
                        )
                        if av_1000_week < 150:
                            st.badge(
                                f"Enter Weighted list size in sidebar", color="yellow"
                            )
                        elif av_1000_week > 85:
                            st.badge(f"100% Access Payment", color="green")
                        elif av_1000_week >= 75 and av_1000_week <= 85:
                            st.badge(f"75% Access Payment", color="yellow")
                        else:
                            st.badge(f"< 75% Access Payment", color="orange")

                # Target Achievement Calculator
                fy_start = pd.Timestamp(2025, 4, 1)
                fy_end = pd.Timestamp(2026, 3, 31)
                last_data_date = pd.Timestamp(filtered_df["appointment_date"].max())

                # Calculate days and weeks left in FY
                days_left_in_fy = (fy_end - last_data_date).days
                weeks_left_in_fy = max(0.0, days_left_in_fy / 7)

                if last_data_date < fy_end:
                    # 1. Calculate Weeks Elapsed and Remaining
                    # Use the start of the FY or the first date in data, whichever is earlier
                    data_start_date = pd.Timestamp(
                        filtered_df["appointment_date"].min()
                    )
                    calc_start_date = min(fy_start, data_start_date)

                    l_date = last_data_date

                    days_elap = (l_date - calc_start_date).days
                    weeks_elapsed = max(0.1, days_elap / 7)

                    days_rem = days_left_in_fy
                    weeks_remaining = max(0.1, weeks_left_in_fy)

                    # 2. ARRS Projection (Respecting Data Lag)
                    # Project from arrs_month till FY end
                    a_end = (
                        arrs_end_date
                        if isinstance(arrs_end_date, pd.Timestamp)
                        else pd.Timestamp(arrs_end_date)
                    )

                    days_from_arrs_to_end = (fy_end - a_end).days
                    weeks_from_arrs_to_end = max(0, days_from_arrs_to_end / 7)
                    projected_future_arrs = (
                        estimated_weekly_arrs * weeks_from_arrs_to_end
                        if "estimated_weekly_arrs" in locals()
                        else 0
                    )

                    # 3. Surgery Baseline Projection
                    # Calculate average weekly surgery appointments from historical data
                    avg_weekly_surgery = len(filtered_df) / weeks_elapsed
                    projected_surgery_baseline = avg_weekly_surgery * weeks_remaining

                    # 4. Annual Target (85 per 1000 per week for 52.14 weeks)
                    annual_target_total = 85 * (list_size / 1000) * 52.14

                    # 5. Total Baseline Projection (Achieved + Future Baseline + Future ARRS)
                    # We assume ARRS continues at the estimated rate for the remainder of the year
                    achieved_so_far = len(filtered_df) + arrs_2526
                    total_baseline_projection = (
                        achieved_so_far
                        + projected_surgery_baseline
                        + projected_future_arrs
                    )

                    # 6. Gap to close
                    gap = annual_target_total - total_baseline_projection

                    # 7. Required additional per week (only for remaining period)
                    required_extra_per_week = (
                        gap / weeks_remaining if weeks_remaining > 0 else 0
                    )

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
                                "Annual Target Total", f"{annual_target_total:.0f}"
                            )
                            st.caption("85 apps/1000/week")
                        with c2:
                            st.metric(
                                "Baseline Projection",
                                f"{total_baseline_projection:.0f}",
                            )
                            st.caption("Historical Avg + ARRS")
                        with c3:
                            color = "normal" if gap <= 0 else "inverse"
                            st.metric(
                                "Gap to Target",
                                f"{max(0, gap):.0f}",
                                delta=f"{gap:.0f}",
                                delta_color=color,
                            )
                            st.caption("Extra apps needed")

                        if gap > 0:
                            st.warning(
                                f":material/target: To reach the annual target, you need to add **{required_extra_per_week:.1f}** additional appointments per week (above our mean weekly surgery apps and ARRS apps) for the remaining **{weeks_remaining:.1f}** weeks."
                            )

                        else:
                            st.success(
                                f":material/done_outline: Based on your current average, you are on track to hit the annual target!"
                            )

                        # 8. Visual Projection Chart
                        st.divider()
                        st.markdown(
                            "#### :material/bar_chart: Annual Projection (Weekly)"
                        )

                        # Create projection dataframe
                        # Historical part
                        proj_df = weekly_agg[["week", "total_appointments"]].copy()
                        proj_df["type"] = "Historical"
                        # ARRS is applied to historical data only if it's within the ARRS period
                        # But for projection, we show the average ARRS contribution
                        proj_df["ARRS"] = arrs_2526 / weeks_elapsed
                        proj_df["Added (Exp)"] = 0
                        proj_df["Catch-up Needed"] = 0

                        # Future part
                        # Fix gap: Start from the next week after the last historical week
                        last_historical_week = weekly_agg["week"].max()
                        future_weeks_list = pd.date_range(
                            start=last_historical_week + pd.Timedelta(weeks=1),
                            end=pd.Timestamp(fy_end),
                            freq="W-MON",
                        )

                        future_data = []
                        for w in future_weeks_list:
                            # The green gap should be reduced by the experimental slider
                            # required_extra_per_week is the TOTAL extra needed per week
                            # exp_add_apps_per_week is what the user is simulating adding
                            remaining_gap = max(
                                0, required_extra_per_week - exp_add_apps_per_week
                            )

                            future_data.append(
                                {
                                    "week": w,
                                    "total_appointments": avg_weekly_surgery,
                                    "type": "Projected Baseline",
                                    "ARRS": estimated_weekly_arrs,
                                    "Added (Exp)": exp_add_apps_per_week,
                                    "Catch-up Needed": remaining_gap,
                                }
                            )

                        if future_data:
                            future_df = pd.DataFrame(future_data)
                            combined_proj_df = pd.concat(
                                [proj_df, future_df], ignore_index=True
                            )

                            fig_proj = px.bar(
                                combined_proj_df,
                                x="week",
                                y=[
                                    "total_appointments",
                                    "Added (Exp)",
                                    "ARRS",
                                    "Catch-up Needed",
                                ],
                                title="Weekly Trajectory to Target",
                                labels={
                                    "value": "Appointments",
                                    "variable": "Component",
                                },
                                color_discrete_map={
                                    "total_appointments": "#0077b6",
                                    "Added (Exp)": "#00b4d8",  # Lighter blue for added apps
                                    "ARRS": "#f48c06",
                                    "Catch-up Needed": "#6a994e",
                                },
                                height=400,
                            )

                            # Add threshold line
                            weekly_threshold = 85 * (list_size / 1000)
                            fig_proj.add_hline(
                                y=weekly_threshold,
                                line_dash="dot",
                                line_color="#c1121f",
                                annotation_text="Weekly Target",
                            )

                            st.plotly_chart(fig_proj, width="stretch")

            else:
                st.warning(
                    "Please select at least one clinician and rota type to display."
                )
        else:
            st.warning(
                "Required columns (appointment_date, clinician, appointment_status, rota_type) not found in the uploaded data."
            )

        with st.expander("Clinician Stats", icon=":material/stethoscope:"):
            st.subheader(":material/stethoscope: Clinician Stats")
            clinician_stats_df = []
            # Use filtered_df_with_dna which includes DNAs for accurate stats
            for clinician in selected_clinicians:
                c_df = filtered_df_with_dna[
                    filtered_df_with_dna["clinician"] == clinician
                ]
                if not c_df.empty:
                    total_apps = len(c_df)
                    c_apps_df = c_df[c_df["appointment_status"] == "Did Not Attend"]
                    dna_count = len(c_apps_df)
                    dna_percentage = (
                        (dna_count / total_apps) * 100 if total_apps > 0 else 0
                    )
                    duration = c_df["duration"].mean()
                    book_to_app = c_df["book_to_app"].mean()

                    data = {
                        "Clinician": clinician,
                        "Total Apps": total_apps,
                        "DNAs": dna_count,
                        "DNA %": round(dna_percentage, 2),
                        "Avg App Duration (mins)": round(duration, 2),
                        "Avg Book to App (mins)": round(book_to_app, 2),
                    }
                    clinician_stats_df.append(data)

            if clinician_stats_df:
                final_df = pd.DataFrame(clinician_stats_df)
                final_df = final_df.sort_values(
                    by="Total Apps", ascending=False
                ).reset_index(drop=True)

                # Add boxp

                # Calculate dynamic height based on number of clinicians
                boxplot_base_height = 200
                height_per_clinician_box = 15
                boxplot_height = boxplot_base_height + (
                    len(selected_clinicians) * height_per_clinician_box
                )

                # Create horizontal boxplot
                fig_duration_box = px.box(
                    filtered_df_with_dna,
                    y="clinician",
                    x="duration",
                    title="Appointment Duration Distribution by Clinician",
                    labels={"clinician": "Clinician", "duration": "Duration (minutes)"},
                    color="clinician",
                    height=boxplot_height,
                    orientation="h",
                )

                fig_duration_box.update_layout(
                    yaxis_title="Clinician",
                    xaxis_title="Duration (minutes)",
                    showlegend=False,
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=0.5,
                        gridcolor="lightgray",
                        range=[0, 150],
                    ),
                    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgray"),
                )

                st.plotly_chart(fig_duration_box, use_container_width=True)

                st.caption(
                    "The boxplot shows the distribution of appointment durations: the box represents the middle 50% of durations, the line inside shows the median, and dots indicate outliers."
                )
                st.divider()
                st.markdown("#### Appointment Stats by Clinician")
                st.dataframe(final_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No clinician data available for the selected filters.")

        # Show file information
        if show_dataframe:
            expander_open = True
        else:
            expander_open = False
        with st.expander(
            "Debug Information", icon=":material/bug_report:", expanded=expander_open
        ):
            if "filtered_df" not in locals():
                st.info(
                    "### :material/info: No Data Selected\nPlease select at least one **Clinician** and **Rota Type** to view debug information and calculations."
                )
            else:
                # ===== DATA STATISTICS SECTION =====
                st.markdown("### :material/bug_report: Debug Info & Data Statistics")
                col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Original Rows:** :orange[{len(combined_df)}]")
            with col2:
                if "filtered_df" in locals():
                    st.markdown(f"**Filtered Rows:** :orange[{len(filtered_df)}]")
                else:
                    st.markdown(f"**Filtered Rows:** :orange[N/A]")
            with col3:
                st.markdown(f"**Columns:** :orange[{len(combined_df.columns)}]")
            with col4:
                st.markdown(f"**Files Loaded:** :orange[{len(file_names)}]")

            # File details with dates
            st.markdown("---")
            st.markdown("**File Details:**")
            total_loaded = 0
            for i, name in enumerate(file_names):
                df_rows = len(dataframes[i])
                total_loaded += df_rows
                # Get date range for this file
                if "appointment_date" in dataframes[i].columns:
                    file_min = pd.to_datetime(
                        dataframes[i]["appointment_date"],
                        format="%d-%b-%y",
                        errors="coerce",
                    ).min()
                    file_max = pd.to_datetime(
                        dataframes[i]["appointment_date"],
                        format="%d-%b-%y",
                        errors="coerce",
                    ).max()
                    if pd.notna(file_min) and pd.notna(file_max):
                        date_range_str = f"{file_min.strftime('%d %b %y')} → {file_max.strftime('%d %b %y')}"
                        st.markdown(
                            f"- `{name}`: :orange[{df_rows}] rows | {date_range_str}"
                        )
                    else:
                        st.markdown(
                            f"- `{name}`: :orange[{df_rows}] rows | :material/error: Date parsing failed"
                        )
                else:
                    st.markdown(f"- `{name}`: :orange[{df_rows}] rows")

            st.markdown(
                f"**Total Loaded:** :orange[{total_loaded}] | **After Cleaning:** :orange[{len(combined_df)}]"
            )

            st.divider()

            # ===== INPUT CONFIGURATION SECTION =====
            st.markdown("#### :material/settings: Input Configuration")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Weighted List Size:** :orange[{list_size}]")
            with col2:
                st.markdown(f"**ARRS Entered:** :orange[{arrs_2526}]")
            with col3:
                status_text = "Yes" if arrs_future else "No"
                st.markdown(f"**Future ARRS:** :orange[{status_text}]")
            with col4:
                status_text = "Yes" if exclude_did_not_attend else "No"
                st.markdown(f"**Exclude DNAs:** :orange[{status_text}]")

            st.divider()

            # ===== DATE RANGE SECTION =====
            st.markdown("#### :material/calendar_check: Date Range")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Check if date_range is defined (it's defined inside the multiselect block)
                d_start_val = date_range[0] if "date_range" in locals() else "N/A"
                st.markdown(f"**Slider Start:** :orange[{d_start_val}]")
            with col2:
                s_end_val = slider_end_date if "slider_end_date" in locals() else "N/A"
                st.markdown(f"**Slider End:** :orange[{s_end_val}]")
            with col3:
                d_start_data = (
                    filtered_df["appointment_date"].min().date()
                    if "filtered_df" in locals()
                    else "N/A"
                )
                st.markdown(f"**Data Start:** :orange[{d_start_data}]")
            with col4:
                d_end_data = (
                    filtered_end_date if "filtered_end_date" in locals() else "N/A"
                )
                st.markdown(f"**Data End:** :orange[{d_end_data}]")

            st.divider()

            # ===== APPOINTMENT COUNTS SECTION =====
            st.markdown("#### :material/call: Appointment Counts")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Filtered Apps:** :orange[{len(filtered_df)}]")
            with col2:
                st.markdown(f"**ARRS Applied:** :orange[{arrs_2526}]")
            with col3:
                st.markdown(f"**Future ARRS Est:** :orange[{future_arrs_apps}]")
            with col4:
                st.markdown(f"**Total + ARRS:** :orange[{total_apps_arrs}]")

            st.markdown("**Formula for Total Apps:**")
            st.code(
                f"total_apps_arrs = total_surgery_apps ({total_surgery_apps}) + arrs_2526 ({arrs_2526}) + future_arrs_apps ({future_arrs_apps}) = {total_apps_arrs}"
            )

            st.divider()

            # ===== TIME CALCULATION SECTION =====
            st.markdown("#### :material/schedule: Time Calculations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Days in Range:** :orange[{time_diff}]")
            with col2:
                st.markdown(f"**Weeks:** :orange[{weeks:.2f}]")
            with col3:
                st.markdown(f"**Months:** :orange[{months:.1f}]")

            st.markdown("**Formula for Weeks:**")
            st.code(f"weeks = time_diff ({time_diff}) / 7 = {weeks:.2f}")

            st.divider()

            # ===== FINAL CALCULATION SECTION =====
            st.markdown("#### :material/edit: Final Calculation")
            st.markdown(
                f"**Formula:** `({total_apps_arrs} ÷ {list_size}) × 1000 ÷ {safe_weeks:.2f}` = :orange[{av_1000_week:.2f}] apps per 1000 per week"
            )

            # Visual breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Total Apps:** :orange[{total_apps_arrs}]")
            with col2:
                st.markdown(f"**List Size:** :orange[{list_size}]")
            with col3:
                st.markdown(f"**Result:** :orange[{av_1000_week:.2f}]")

            st.markdown("**Detailed Formula:**")
            st.code(
                f"av_1000_week = (total_apps_arrs ({total_apps_arrs}) / list_size ({list_size})) * 1000 / safe_weeks ({safe_weeks:.2f}) = {av_1000_week:.2f}"
            )

            st.divider()

            # ===== FUTURE PREDICTION SECTION =====
            st.markdown(
                "#### :material/online_prediction: Future Prediction (Target Calculator)"
            )
            if "weeks_remaining" in locals():
                st.markdown(f"**Days Left in FY:** :orange[{days_left_in_fy}]")
                st.markdown(f"**Weeks Left in FY:** :orange[{weeks_left_in_fy:.2f}]")
                st.markdown("**Formula for Future Baseline:**")
                st.code(
                    f"projected_surgery_baseline = avg_weekly_surgery ({avg_weekly_surgery:.2f}) * weeks_remaining ({weeks_remaining:.2f}) = {projected_surgery_baseline:.0f}"
                )
                st.markdown("**Formula for Future ARRS:**")
                st.code(
                    f"projected_future_arrs = estimated_weekly_arrs ({estimated_weekly_arrs:.2f}) * weeks_from_arrs_to_end ({weeks_from_arrs_to_end:.2f}) = {projected_future_arrs:.0f}"
                )
                st.markdown("**Total Baseline Projection:**")
                st.code(
                    f"total_baseline_projection = achieved_so_far ({achieved_so_far}) + projected_surgery_baseline ({projected_surgery_baseline:.0f}) + projected_future_arrs ({projected_future_arrs:.0f}) = {total_baseline_projection:.0f}"
                )
                st.markdown("**Gap to Target:**")
                st.code(
                    f"gap = annual_target_total ({annual_target_total:.0f}) - total_baseline_projection ({total_baseline_projection:.0f}) = {gap:.0f}"
                )
            else:
                st.info(
                    "Target Calculator data not available (data might extend beyond FY end)."
                )

            if show_dataframe:
                st.divider()
                st.markdown("#### Dataframes")
                st.write("**filtered_df**")
                st.dataframe(filtered_df, width="stretch")
                st.write("**weekly_agg**")
                st.dataframe(weekly_agg, width="stretch")
                st.write("**monthly_agg**")
                st.dataframe(monthly_agg, width="stretch")
                st.caption("Access ES Tracker")

        # Download combined CSV
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Combined CSV",
            data=csv,
            file_name="combined_data.csv",
            mime="text/csv",
        )

else:
    st.info(
        """### :material/line_start_arrow:  Please upload CSV files using the sidebar to get started!"""
    )
    st.info(
        """Create an **appointment report** in clincal reporting on SytmOne for the time period required. **Breakdown** search with the following columns:  
            
            - Appointment Date  
            - Appointment duration (actual)
            - Appointment Status  
            - Rota type
            - Appointment flags  
            - Clinician  
            - Appointment status 
            - Time between booking and appointment 
            - Patient ID  
            
Export as multiple searhes (incrementing date with each) as SystmOne only allow and export of 30 000 rows per search.  
Download the search below and input into SystmOne.  
Any questions please email jan.duplessis@nhs.net"""
    )
