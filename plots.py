"""
Plotting functions for Access ES Appointment Tracker
Contains all Plotly visualization functions
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from config import *


def create_scatter_plot(df, view_option, selected_clinicians):
    """Create scatter plot based on selected view option"""
    # Calculate dynamic height
    if view_option == "Rota/Flags":
        num_categories = df['appointment_flags'].nunique()
    else:
        num_categories = len(selected_clinicians)
    
    fig_height = BASE_PLOT_HEIGHT + (num_categories * HEIGHT_PER_CLINICIAN)
    
    if view_option == "Rota Type":
        fig = px.strip(
            df,
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
    
    elif view_option == "App Flags":
        fig = px.strip(
            df,
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
    
    elif view_option == "DNAs":
        plot_df = df.copy()
        status_order = {'Finished': 0, 'Did Not Attend': 1}
        plot_df['_status_sort'] = plot_df['appointment_status'].map(status_order)
        plot_df = plot_df.sort_values('_status_sort')
        
        fig = px.strip(
            plot_df,
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
        fig.update_traces(marker=dict(opacity=0.6))
    
    elif view_option == "Rota/Flags":
        fig = px.strip(
            df,
            x='appointment_date',
            y='appointment_flags',
            color='rota_type',
            title="Appointments by Date and Flags (Colored by Rota Type)",
            labels={
                'appointment_date': 'Appointment Date',
                'appointment_flags': 'Appointment Flags',
                'rota_type': 'Rota Type'
            },
            height=fig_height
        )
    
    fig.update_layout(
        xaxis_title="Appointment Date",
        yaxis_title="Clinician" if view_option != "Rota/Flags" else "Appointment Flags",
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        )
    )
    
    return fig


def create_weekly_trend_plot(weekly_agg, show_per_1000, list_size, arrs_end_date, should_apply_arrs):
    """Create weekly trend plot (bar or line based on toggle)"""
    weekly_agg_plot = weekly_agg.copy()
    
    if show_per_1000:
        # Line chart for per 1000 view
        y_plot = "per_1000"
        threshold_plot = THRESHOLD_100_PERCENT
        plot_title = "Appointments per 1000 pts per Week"
        
        fig = px.line(
            weekly_agg_plot,
            x='week',
            y=y_plot,
            title=plot_title,
            labels={
                'week': 'Week Starting Date',
                y_plot: 'Apps per 1000 per week'
            },
            height=WEEKLY_PLOT_HEIGHT,
            markers=False,
            color_discrete_sequence=[PLOT_COLORS['actual_appointments']]
        )
        
        fig.update_traces(
            name="Actual Apps per 1000",
            showlegend=True,
            text=weekly_agg_plot[y_plot].round(2),
            textposition='top center',
            mode='lines+text',
            textfont=dict(size=10)
        )
        
        # Add ARRS line
        fig.add_scatter(
            x=weekly_agg_plot['week'],
            y=weekly_agg_plot['per_1000_with_arrs'],
            mode='lines+text',
            name='Apps per 1000 + ARRS',
            line=dict(dash='solid', color=PLOT_COLORS['arrs_historical']),
            text=weekly_agg_plot['per_1000_with_arrs'].round(2),
            textposition='top center',
            textfont=dict(size=9)
        )
        
        # Add mean line
        mean_val = weekly_agg_plot['per_1000_with_arrs'].mean()
        fig.add_hline(
            y=mean_val,
            line_dash='dot',
            line_color=PLOT_COLORS['catchup_needed'],
            line_width=2,
            annotation_text=f'Mean Apps + ARRS ({mean_val:.2f})',
            annotation_position='top right'
        )
    else:
        # Bar chart for total appointments
        y_plot = "total_appointments"
        threshold_plot = THRESHOLD_100_PERCENT * (list_size / 1000)
        plot_title = "Total Appointments per Week"
        
        fig = px.bar(
            weekly_agg_plot,
            x='week',
            y=[y_plot, 'arrs_historical', 'arrs_future'],
            title=plot_title,
            labels={
                'week': 'Week Starting Date',
                'value': 'Count',
                'variable': 'Type'
            },
            height=WEEKLY_PLOT_HEIGHT,
            color_discrete_map={
                y_plot: PLOT_COLORS['actual_appointments'],
                'arrs_historical': PLOT_COLORS['arrs_historical'],
                'arrs_future': PLOT_COLORS['arrs_future']
            }
        )
        
        # Add text labels showing cumulative totals
        fig.update_traces(
            selector=dict(name=y_plot),
            text=weekly_agg_plot[y_plot],
            textposition='inside'
        )
        fig.update_traces(
            selector=dict(name='arrs_historical'),
            text=(weekly_agg_plot[y_plot] + weekly_agg_plot['arrs_historical']).round(0),
            textposition='inside'
        )
        fig.update_traces(
            selector=dict(name='arrs_future'),
            text=weekly_agg_plot['total_with_arrs'].round(0),
            textposition='inside'
        )
        
        # Update legend names
        newnames = {
            y_plot: 'Actual Appointments',
            'arrs_historical': 'ARRS (Historical)',
            'arrs_future': 'ARRS (Future Est)'
        }
        fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
    
    # Add threshold line
    fig.add_hline(
        y=threshold_plot,
        line_dash='dash' if show_per_1000 else 'dot',
        line_color=PLOT_COLORS['threshold_line'],
        line_width=2,
        annotation_text=f'Threshold ({threshold_plot:.2f})',
        annotation_position='top right'
    )
    
    # Add ARRS cutoff line
    if should_apply_arrs:
        fig.add_vline(
            x=arrs_end_date,
            line_dash='dashdot',
            line_color=PLOT_COLORS['arrs_cutoff_line'],
            line_width=2
        )
        fig.add_annotation(
            x=arrs_end_date,
            text='ARRS Prediction Start',
            showarrow=False,
            xanchor='right',
            yanchor='top'
        )
    
    fig.update_layout(
        xaxis_title="Week Starting Date",
        yaxis_title="Total Appointments",
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        )
    )
    
    return fig


def create_monthly_trend_plot(monthly_agg, arrs_end_date, should_apply_arrs):
    """Create monthly trend stacked bar plot"""
    monthly_agg_plot = monthly_agg.copy()
    avg_monthly_appointments = monthly_agg['total_appointments'].mean()
    
    fig = px.bar(
        monthly_agg_plot,
        x='month',
        y=['total_appointments', 'arrs_historical', 'arrs_future'],
        title='Total Appointments per Month',
        labels={
            'month': 'Month',
            'value': 'Count',
            'variable': 'Type'
        },
        height=MONTHLY_PLOT_HEIGHT,
        color_discrete_map={
            'total_appointments': PLOT_COLORS['actual_appointments'],
            'arrs_historical': PLOT_COLORS['arrs_historical'],
            'arrs_future': PLOT_COLORS['arrs_future']
        }
    )
    
    # Add text labels showing cumulative totals
    fig.update_traces(
        selector=dict(name='total_appointments'),
        text=monthly_agg_plot['total_appointments'],
        textposition='inside'
    )
    fig.update_traces(
        selector=dict(name='arrs_historical'),
        text=(monthly_agg_plot['total_appointments'] + monthly_agg_plot['arrs_historical']).round(0),
        textposition='inside'
    )
    fig.update_traces(
        selector=dict(name='arrs_future'),
        text=monthly_agg_plot['total_with_arrs'].round(0),
        textposition='inside'
    )
    
    # Update legend names
    newnames = {
        'total_appointments': 'Actual Appointments',
        'arrs_historical': 'ARRS (Historical)',
        'arrs_future': 'ARRS (Future Est)'
    }
    fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
    
    # Add average line
    fig.add_hline(
        y=avg_monthly_appointments,
        line_dash='dot',
        line_color=PLOT_COLORS['average_line'],
        line_width=2,
        annotation_text=f'Monthly Average Completed Apps ({avg_monthly_appointments:.1f})',
        annotation_position='top left'
    )
    
    # Add ARRS cutoff line
    if should_apply_arrs:
        fig.add_vline(
            x=arrs_end_date,
            line_dash='dashdot',
            line_color=PLOT_COLORS['arrs_cutoff_line'],
            line_width=2
        )
        fig.add_annotation(
            x=arrs_end_date,
            text='ARRS Prediction Start',
            showarrow=False,
            xanchor='right',
            yanchor='top'
        )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Appointments",
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        )
    )
    
    return fig


def create_duration_boxplot(df, selected_clinicians):
    """Create boxplot for appointment duration by clinician"""
    boxplot_height = BOXPLOT_BASE_HEIGHT + (len(selected_clinicians) * HEIGHT_PER_CLINICIAN)
    
    fig = px.box(
        df,
        y='clinician',
        x='duration',
        title='Appointment Duration Distribution by Clinician',
        labels={
            'clinician': 'Clinician',
            'duration': 'Duration (minutes)'
        },
        color='clinician',
        height=boxplot_height,
        orientation='h'
    )
    
    fig.update_layout(
        yaxis_title="Clinician",
        xaxis_title="Duration (minutes)",
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            range=[0, 150]
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        )
    )
    
    return fig


def create_projection_chart(combined_proj_df, list_size):
    """Create weekly projection chart for target achievement"""
    fig = px.bar(
        combined_proj_df,
        x='week',
        y=['total_appointments', 'Added (Exp)', 'ARRS', 'Catch-up Needed'],
        title="Weekly Trajectory to Target",
        labels={'value': 'Appointments', 'variable': 'Component'},
        color_discrete_map={
            'total_appointments': PLOT_COLORS['actual_appointments'],
            'Added (Exp)': PLOT_COLORS['forecasted_apps'],
            'ARRS': PLOT_COLORS['arrs_historical'],
            'Catch-up Needed': PLOT_COLORS['catchup_needed']
        },
        height=PROJECTION_PLOT_HEIGHT
    )
    
    # Add text labels showing cumulative totals
    fig.update_traces(
        selector=dict(name='total_appointments'),
        text=combined_proj_df['total_appointments'].round(0),
        textposition='inside'
    )
    fig.update_traces(
        selector=dict(name='Added (Exp)'),
        text=(combined_proj_df['total_appointments'] + combined_proj_df['Added (Exp)']).round(0),
        textposition='inside'
    )
    fig.update_traces(
        selector=dict(name='ARRS'),
        text=(combined_proj_df['total_appointments'] + combined_proj_df['Added (Exp)'] + combined_proj_df['ARRS']).round(0),
        textposition='inside'
    )
    fig.update_traces(
        selector=dict(name='Catch-up Needed'),
        text=(combined_proj_df['total_appointments'] + combined_proj_df['Added (Exp)'] + combined_proj_df['ARRS'] + combined_proj_df['Catch-up Needed']).round(0),
        textposition='inside'
    )
    
    # Add threshold line
    weekly_threshold = THRESHOLD_100_PERCENT * (list_size / 1000)
    fig.add_hline(
        y=weekly_threshold,
        line_dash='dot',
        line_color=PLOT_COLORS['threshold_line'],
        annotation_text="Weekly Target"
    )
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Appointments",
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray'
        )
    )
    
    return fig


def create_clinician_stats_histograms(stats_df):
    """Create 1x4 subplot of histograms using seaborn for clinician statistics"""
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(18, 3))
    
    # Total Apps histogram
    sns.histplot(
        data=stats_df,
        x='Total Apps',
        ax=axes[0],
        color='#1f77b4',
        kde=True,
        bins=15
    )
    axes[0].set_title('Total Apps Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Total Apps', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    
    # DNAs histogram
    sns.histplot(
        data=stats_df,
        x='DNAs',
        ax=axes[1],
        color='#d62728',
        kde=True,
        bins=15
    )
    axes[1].set_title('DNAs Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('DNAs', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    
    # Avg App Duration histogram
    sns.histplot(
        data=stats_df,
        x='Avg App Duration (mins)',
        ax=axes[2],
        color='#79803b',
        kde=True,
        bins=15
    )
    axes[2].set_title('Avg App Duration (mins) Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Duration (mins)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    
    # Avg Book to App histogram
    sns.histplot(
        data=stats_df,
        x='Avg Book to App (mins)',
        ax=axes[3],
        color='#dd8259',
        kde=True,
        bins=15
    )
    axes[3].set_title('Avg Book to App (mins) Distribution', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Book to App (mins)', fontsize=10)
    axes[3].set_ylabel('Frequency', fontsize=10)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[3].spines['left'].set_visible(False)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig
