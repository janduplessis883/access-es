import streamlit as st
import pandas as pd
import plotly.express as px
import io
from datetime import date

st.set_page_config(page_title="Access ES Tracker", layout="wide", page_icon=':material/child_hat:')

st.title("Access ES Tracker")
st.markdown("Upload multiple CSV files to merge them into a single dataframe")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload CSV Files")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Select one or more CSV files to combine"
    )

    st.divider()

    list_size = st.number_input(
        "List Size",
        min_value=1,
        value=10,
        step=1,
        help="Set the number of items to display"
    )

    exclude_did_not_attend = st.checkbox(
        "Exclude 'Did Not Attend'",
        value=True,
        help="Filter out appointments with 'Did Not Attend' status"
    )

    st.divider()

    st.header("ARRS 25/26")
    arrs_2526 = st.number_input("Total ARRS to date.", min_value=0, value=0, step=1)

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
        st.badge(f"Enter ICB ARRS total value to date", color='red')
        arrs_future = False
        arrs_month = "April 25"
        arrs_end_date = arrs_month_map[arrs_month]
    else:
        arrs_month = st.selectbox("Select month for ARRS input", options=[
            "April 25", "May 25", "June 25", "July 25", "August 25", "September 25", "October 25", "November 25", "December 25", "January 26", "February 26", "March 26"], index=0, help="Select the month corresponding to the ARRS input.")

        arrs_end_date = arrs_month_map[arrs_month]
        arrs_start_date = date(2025, 4, 1)
        arrs_weeks_span = (arrs_end_date - arrs_start_date).days / 7

        arrs_future = st.checkbox('Estimate Future ARRS?', value=False, help="Estimate future ARRS based on current weekly rate")
        if arrs_future:
            estimated_weekly_arrs = arrs_2526 / arrs_weeks_span
            st.info(f"Future ARRS estimation at **{estimated_weekly_arrs:.0f}** ARRS apps per week - {arrs_weeks_span:.1f} weeks to {arrs_month}.")
        else:
            st.error(f":material/warning: Move your date slider to the **last day of {arrs_month}**")
            st.error(f":material/warning: ARRS **will not be applied** to future months! Your Average Apps per 1000 per week will be underestimated.")
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
            st.sidebar.success(f"✓ Loaded: {uploaded_file.name}")
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
                # If that fails, try auto-detection
                combined_df['appointment_date'] = pd.to_datetime(
                    combined_df['appointment_date'],
                    format='mixed',
                    dayfirst=False
                )

        # Scatter plot section
        st.subheader("Filters and Visualization")

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
                if exclude_did_not_attend:
                    filtered_df = filtered_df[filtered_df['appointment_status'] != 'Did Not Attend']

                # Determine if ARRS should be applied based on filtered end date (needed for chart annotations)
                filtered_end_date = filtered_df['appointment_date'].max().date()
                should_apply_arrs = filtered_end_date >= arrs_end_date and arrs_2526 > 0

                # Create weekly aggregation for all appointments in filtered dataframe
                weekly_df = filtered_df.copy()
                weekly_df['week'] = weekly_df['appointment_date'].dt.to_period('W').dt.start_time
                weekly_agg = weekly_df.groupby('week').size().reset_index(name='total_appointments')

                # Create monthly aggregation for all appointments in filtered dataframe
                monthly_df = filtered_df.copy()
                monthly_df['month'] = monthly_df['appointment_date'].dt.to_period('M').dt.start_time
                monthly_agg = monthly_df.groupby('month').size().reset_index(name='total_appointments')

                # Toggle to select hue
                use_rota_type = st.toggle(
                    "Color by Rota Type",
                    value=True,
                    help="Toggle ON to color by Rota Type (default), OFF to color by Appointment Status"
                )

                hue_column = 'rota_type' if use_rota_type else 'appointment_status'

                # Sort dataframe for better visibility when coloring by appointment_status
                if hue_column == 'appointment_status':
                    # Sort so 'Finished' appointments are plotted first, 'Did Not Attend' on top
                    status_order = {'Finished': 0, 'Did Not Attend': 1}
                    plot_df = filtered_df.copy()
                    plot_df['_status_sort'] = plot_df['appointment_status'].map(status_order)
                    plot_df = plot_df.sort_values('_status_sort')
                else:
                    plot_df = filtered_df

                # Calculate dynamic height based on number of clinicians
                base_height = 300
                height_per_clinician = 15
                fig_height = base_height + (len(selected_clinicians) * height_per_clinician)

                # Create scatter plot
                fig = px.scatter(
                    plot_df,
                    x='appointment_date',
                    y='clinician',
                    color=hue_column,
                    title=f"Appointments by Date and Clinician (Colored by {hue_column.replace('_', ' ').title()})",
                    labels={
                        'appointment_date': 'Appointment Date',
                        'clinician': 'Clinician',
                        hue_column: hue_column.replace('_', ' ').title()
                    },
                    height=fig_height
                )

                fig.update_layout(
                    xaxis_title="Appointment Date",
                    yaxis_title="Clinician",
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

                # Add threshold line to weekly plot
                fig_weekly.add_hline(
                    y=threshold,
                    line_dash='dash',
                    line_color='red',
                    annotation_text=f'Threshold ({threshold:.2f})',
                    annotation_position='right'
                )

                # Add ARRS end date vertical line if applicable
                if should_apply_arrs:
                    fig_weekly.add_vline(
                        x=pd.Timestamp(arrs_end_date),
                        line_dash='dash',
                        line_color='blue'
                    )
                    fig_weekly.add_annotation(
                        x=pd.Timestamp(arrs_end_date),
                        text='ARRS End Date',
                        showarrow=False,
                        xanchor='left',
                        yanchor='top'
                    )

                # Add estimated ARRS indicator if applicable
                if arrs_future and should_apply_arrs:
                    # Add yellow horizontal band showing estimated weekly ARRS rate
                    fig_weekly.add_hline(
                        y=estimated_weekly_arrs,
                        line_dash='dash',
                        line_color='orange',
                        annotation_text=f'Est. Weekly ARRS ({estimated_weekly_arrs:.0f})',
                        annotation_position='right'
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

                # Add ARRS end date vertical line if applicable
                if should_apply_arrs:
                    fig_monthly.add_vline(
                        x=pd.Timestamp(arrs_end_date),
                        line_dash='dash',
                        line_color='blue'
                    )
                    fig_monthly.add_annotation(
                        x=pd.Timestamp(arrs_end_date),
                        text='ARRS End Date',
                        showarrow=False,
                        xanchor='left',
                        yanchor='top'
                    )

                # Add estimated ARRS indicator if applicable
                if arrs_future and should_apply_arrs:
                    # Add yellow horizontal band showing estimated weekly ARRS rate
                    fig_monthly.add_hline(
                        y=estimated_weekly_arrs,
                        line_dash='dash',
                        line_color='orange',
                        annotation_text=f'Est. Weekly ARRS ({estimated_weekly_arrs:.0f})',
                        annotation_position='right'
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
                time_diff = (filtered_df['appointment_date'].max() - filtered_df['appointment_date'].min()).days
                weeks = time_diff / 7
                months = time_diff / 30.44

                # Determine if ARRS should be applied based on filtered end date
                filtered_end_date = filtered_df['appointment_date'].max().date()
                should_apply_arrs = filtered_end_date >= arrs_end_date and arrs_2526 > 0

                if arrs_future and should_apply_arrs:
                    # Calculate weeks from ARRS end date to filtered end date for future estimation
                    future_weeks = (filtered_end_date - arrs_end_date).days / 7
                    future_arrs_apps = int(round(estimated_weekly_arrs * future_weeks, 0))
                    total_apps_arrs = len(filtered_df) + arrs_2526 + future_arrs_apps
                elif should_apply_arrs:
                    total_apps_arrs = len(filtered_df) + arrs_2526
                else:
                    total_apps_arrs = len(filtered_df)

                av_1000_week = (total_apps_arrs / list_size) * 1000 / weeks if weeks > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                start_date = filtered_df['appointment_date'].min().strftime('%d %b %y')
                end_date = filtered_df['appointment_date'].max().strftime('%d %b %y')
                with col1:
                    st.metric("Total Surgery Appointments", len(filtered_df))
                with col2:
                    if should_apply_arrs:
                        if arrs_future:
                            total_arrs_estimated = arrs_2526 + future_arrs_apps
                            st.metric(f"Total ARRS estimated to {end_date}", total_arrs_estimated)
                            st.badge(f"+ Future ARRS Applied! ({arrs_2526} + {future_arrs_apps})", color='red')
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
                    elif av_1000_week >= 84.5 and av_1000_week <= 200:
                        st.badge(f"100% Access Payment", color='green')
                    elif av_1000_week >= 75 and av_1000_week < 84.5:
                        st.badge(f"75% Access Payment", color='yellow')
                    else:
                        st.badge(f"< 75% Access Payment", color='orange')



            else:
                st.warning("Please select at least one clinician and rota type to display.")
        else:
            st.warning("Required columns (appointment_date, clinician, appointment_status, rota_type) not found in the uploaded data.")

        # Display total data statistics
        st.markdown("**Total Data Statistics** (Original Dataframe)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.badge(f"Total Rows (All Data) **{len(combined_df)}**", color="blue")
        with col2:
            st.badge(f"Total Columns (All Data) **{len(combined_df.columns)}**", color="blue")
        with col3:
            st.badge(f"Total Files **{len(file_names)}** ", color="blue")

        # Show file information
        with st.expander("File Information"):
            for i, name in enumerate(file_names):
                st.write(f"**File {i+1}**: {name} ({len(dataframes[i])} rows)")

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
