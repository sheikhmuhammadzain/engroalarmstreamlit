import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
import plotly.io as pio
import textwrap
import os
import io
import contextlib
import traceback
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False
warnings.filterwarnings('ignore')

# Global theming for Plotly: white background with green accents
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#16A34A",  # Green 600
    "#22C55E",  # Green 500
    "#059669",  # Emerald 600
    "#10B981",  # Emerald 500
    "#065F46",  # Green 800
    "#6EE7B7",  # Emerald 300
    "#34D399"   # Emerald 400
]

# Page configuration
st.set_page_config(
    page_title="Alarm Data Analytics Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #FFFFFF;
    }
    /* Metric cards */
    .stMetric {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E5E7EB; /* gray-200 */
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        border-left: 4px solid #16A34A; /* primary green accent */
    }
    /* Equalize metric card heights */
    .stMetric, div[data-testid="stMetric"] {
        height: 130px;
        min-height: 130px;
    }
    div[data-testid="stMetric"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    /* Ensure column children stretch */
    div[data-testid="column"] > div {
        height: 100%;
    }
    /* Improve section dividers */
    hr, .stMarkdown hr { border-color: #E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚨 Alarm System Analytics Dashboard")
st.markdown("### Industrial Control System Event Analysis")

# Sidebar for file upload and filters
with st.sidebar:
    st.header("📁 Data Input")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload cleaned CSV file",
        type=['csv'],
        help="Upload the cleaned alarm data CSV file"
    )
    
    # Use example data option
    use_example = st.checkbox("Use example data", value=True if not uploaded_file else False)
    
    if uploaded_file is not None or use_example:
        st.success("✅ Data loaded successfully!")
    
    st.markdown("---")

# Load data function
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Create example data if no file uploaded
        st.info("Using example data. Upload your own CSV file to see your actual data.")
        # This would be your actual data loading
        try:
            df = pd.read_csv('01feb_cleaned.csv')
        except:
            st.error("Please upload a CSV file or ensure '01feb_cleaned.csv' is in the directory")
            return None
    
    # Data preparation
    df['Event Time'] = pd.to_datetime(df['Event Time'])
    df['Hour'] = df['Event Time'].dt.hour
    df['Minute'] = df['Event Time'].dt.minute
    df['Date'] = df['Event Time'].dt.date
    df['Action'] = df['Action'].fillna('NO ACTION')
    
    # Create alarm categories
    alarm_conditions = ['ALARM', 'PVHIHI', 'PVHI', 'PVLO', 'PVLOW', 'PVLOLOW', 
                       'HIHI', 'HI', 'LO', 'LOLO', 'FAIL']
    df['Is_Alarm'] = df['Condition'].str.upper().str.contains(
        '|'.join(alarm_conditions), na=False
    )
    
    return df

# ---------- AI helpers ----------
def build_ai_context(df: pd.DataFrame, max_numeric_cols: int = 6, max_cat_cols: int = 6, sample_rows: int = 5) -> str:
    """Build a concise text context describing the filtered dataframe.
    Limits numeric/categorical summaries and includes a few sample rows to keep token usage low.
    """
    if df is None or len(df) == 0:
        return "No data available."

    lines = []
    lines.append("DATAFRAME OVERVIEW")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")
    lines.append("COLUMNS (name: dtype, non-null count):")
    for col in list(df.columns)[:50]:
        dtype = str(df[col].dtype)
        nonnull = int(df[col].notna().sum())
        lines.append(f"- {col}: {dtype}, non-null={nonnull:,}")

    # Numeric summaries
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        lines.append("")
        lines.append("NUMERIC SUMMARIES (mean, std, min, max):")
        for col in num_cols[:max_numeric_cols]:
            desc = df[col].describe()
            try:
                mean = float(desc.get('mean', 0))
                std = float(desc.get('std', 0))
                min_v = float(desc.get('min', 0))
                max_v = float(desc.get('max', 0))
                lines.append(f"- {col}: mean={mean:.2f}, std={std:.2f}, min={min_v:.2f}, max={max_v:.2f}")
            except Exception:
                pass

    # Categorical top values
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        lines.append("")
        lines.append("CATEGORICAL TOP VALUES (top 5):")
        for col in cat_cols[:max_cat_cols]:
            vc = df[col].value_counts(dropna=True).head(5)
            compact = ", ".join([f"{idx} ({cnt})" for idx, cnt in vc.items()])
            lines.append(f"- {col}: {compact}")

    # Sample rows
    try:
        n = min(sample_rows, len(df))
        sample = df.head(n)
        lines.append("")
        lines.append(f"SAMPLE ROWS (first {n}):")
        # Keep sample compact by converting datetimes to string
        sample_cp = sample.copy()
        for c in sample_cp.columns:
            if np.issubdtype(sample_cp[c].dtype, np.datetime64):
                sample_cp[c] = sample_cp[c].dt.strftime('%Y-%m-%d %H:%M:%S')
        lines.append(sample_cp.to_csv(index=False))
    except Exception:
        pass

    return "\n".join(lines)


def ask_openai(question: str, context: str, model: str = "gpt-4o-mini", code_mode: bool = False) -> str:
    """Ask OpenAI using a chat completion. Expects API key in env or session state.
    If code_mode is True, the assistant should return a single Python fenced code block
    that sets a variable named `result` (and optionally `fig` for Plotly).
    """
    if not _OPENAI_AVAILABLE:
        return "OpenAI Python package is not installed. Please run: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        return "Missing OPENAI_API_KEY. Add it to environment or enter it in the field above."

    try:
        client = OpenAI(api_key=api_key)
        if code_mode:
            system_prompt = (
                "You are a helpful data analyst writing Python to analyze a pandas DataFrame named df. "
                "Use ONLY the provided context to infer column names/types and avoid external I/O or network access. "
                "Return a single fenced Python code block only (no prose), which: "
                "1) computes the answer using the DataFrame df, "
                "2) assigns the main output to a variable named result (DataFrame/Series/scalar), and "
                "3) optionally assigns a Plotly figure to a variable named fig. "
                "Do not import modules. Use provided pd/np/px if needed."
            )
            user_prompt = (
                f"Context about df:\n\n{context}\n\n"
                f"Task: Write Python code to answer the user's question on df. Question: {question}"
            )
        else:
            system_prompt = (
                "You are a helpful data analyst. Use ONLY the provided context to answer the user's question. "
                "If the answer is not derivable from the context, say 'I cannot determine from the provided data.' "
                "When relevant, include concise reasoning and, if helpful, short pandas code snippets to reproduce the answer."
            )
            user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI request failed: {e}"


def extract_python_code(text: str) -> str:
    """Extract the first Python fenced code block from the LLM text."""
    if not text:
        return ""
    fences = [
        ("```python", "```"),
        ("```py", "```"),
        ("```", "```"),
    ]
    for start, end in fences:
        if start in text:
            start_idx = text.find(start) + len(start)
            end_idx = text.find(end, start_idx)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()
    return ""


def run_user_code(code: str, df: pd.DataFrame):
    """Execute user-provided code in a restricted environment.
    Exposes df, pd, np, px, go. Requires code to optionally set `result` and/or `fig`.
    Returns (env, stdout_text, error_text).
    """
    safe_builtins = {
        'len': len, 'range': range, 'min': min, 'max': max, 'sum': sum,
        'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'abs': abs, 'round': round,
        'print': print, 'any': any, 'all': all, 'map': map, 'filter': filter
    }
    g = {
        '__builtins__': safe_builtins,
        'pd': pd, 'np': np, 'px': px, 'go': go,
    }
    l = {'df': df}
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, g, l)
        return l, stdout_buf.getvalue(), None
    except Exception:
        tb = traceback.format_exc(limit=2)
        return l, stdout_buf.getvalue(), tb

# Main app logic
if uploaded_file is not None or use_example:
    # Load data
    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        df = load_data(None)
    
    if df is not None:
        # Sidebar filters
        with st.sidebar:
            st.header("🎛️ Filters")
            
            # Date range filter
            date_range = st.date_input(
                "Select date range",
                value=(df['Event Time'].min(), df['Event Time'].max()),
                min_value=df['Event Time'].min(),
                max_value=df['Event Time'].max()
            )
            
            # Source filter
            sources = st.multiselect(
                "Select sources",
                options=df['Source'].unique(),
                default=None
            )
            
            # Condition filter
            conditions = st.multiselect(
                "Select conditions",
                options=df['Condition'].unique(),
                default=None
            )
            
            # Alarm only filter
            show_alarms_only = st.checkbox("Show alarms only", value=False)
        
        # Apply filters
        filtered_df = df.copy()
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= date_range[0]) & 
                (filtered_df['Date'] <= date_range[1])
            ]
        
        if sources:
            filtered_df = filtered_df[filtered_df['Source'].isin(sources)]
        
        if conditions:
            filtered_df = filtered_df[filtered_df['Condition'].isin(conditions)]
        
        if show_alarms_only:
            filtered_df = filtered_df[filtered_df['Is_Alarm'] == True]
        
        # Main dashboard
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🚨 Critical Events", 
            "📈 Process Variables",
            "📋 Data Table",
            "📑 Summary Report",
            "🤖 Ask AI"
        ])
        
        # Tab 1: Overview
        with tab1:
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Events", f"{len(filtered_df):,}")
            
            with col2:
                alarm_count = filtered_df['Is_Alarm'].sum()
                st.metric("Total Alarms", f"{alarm_count:,}", 
                         f"{alarm_count/len(filtered_df)*100:.1f}%")
            
            with col3:
                ack_count = filtered_df['Action'].str.contains('ACK', na=False).sum()
                st.metric("Acknowledged", f"{ack_count:,}",
                         f"{ack_count/len(filtered_df)*100:.1f}%")
            
            with col4:
                unique_sources = filtered_df['Source'].nunique()
                st.metric("Unique Sources", f"{unique_sources:,}")
            
            with col5:
                value_count = filtered_df['Value'].notna().sum()
                st.metric("Events with Values", f"{value_count:,}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Events over time
                st.subheader("📈 Events Over Time")
                time_series = filtered_df.groupby(
                    filtered_df['Event Time'].dt.floor('5min')
                ).size().reset_index(name='count')
                
                fig_timeline = px.line(time_series, x='Event Time', y='count',
                                       title="Event Frequency (5-min intervals)")
                fig_timeline.update_layout(height=300)
                st.plotly_chart(fig_timeline, width='stretch')
                
                # Top sources
                st.subheader("🔝 Top Event Sources")
                top_sources = filtered_df['Source'].value_counts().head(10)
                fig_sources = px.bar(x=top_sources.values, y=top_sources.index,
                                     orientation='h', title="Top 10 Sources")
                fig_sources.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_sources, width='stretch')
            
            with col2:
                # Condition distribution
                st.subheader("⚠️ Condition Types")
                condition_counts = filtered_df['Condition'].value_counts().head(10)
                fig_conditions = px.pie(values=condition_counts.values, 
                                       names=condition_counts.index,
                                       title="Top 10 Conditions")
                fig_conditions.update_layout(height=300)
                st.plotly_chart(fig_conditions, width='stretch')
                
                # Hourly distribution
                st.subheader("⏰ Hourly Distribution")
                hourly_dist = filtered_df['Hour'].value_counts().sort_index()
                fig_hourly = px.bar(x=hourly_dist.index, y=hourly_dist.values,
                                   title="Events by Hour of Day")
                fig_hourly.update_layout(height=400, xaxis_title="Hour", yaxis_title="Count")
                st.plotly_chart(fig_hourly, width='stretch')
            
            # Location analysis
            st.subheader("📍 Location Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                top_locations = filtered_df['Location Tag'].value_counts().head(10)
                fig_locations = px.bar(x=top_locations.index, y=top_locations.values,
                                      title="Top 10 Locations")
                fig_locations.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_locations, width='stretch')
            
            with col2:
                # Action types
                action_counts = filtered_df['Action'].value_counts()
                fig_actions = px.pie(values=action_counts.values, 
                                    names=action_counts.index,
                                    title="Action Types Distribution")
                st.plotly_chart(fig_actions, width='stretch')
        
        # Tab 2: Critical Events
        with tab2:
            st.header("🚨 Critical Events Analysis")
            
            alarm_df = filtered_df[filtered_df['Is_Alarm']]
            
            if len(alarm_df) > 0:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Alarms", f"{len(alarm_df):,}")
                
                with col2:
                    critical_sources = alarm_df['Source'].nunique()
                    st.metric("Sources with Alarms", f"{critical_sources}")
                
                with col3:
                    ack_alarms = alarm_df['Action'].str.contains('ACK', na=False).sum()
                    st.metric("Acknowledged Alarms", f"{ack_alarms:,}")
                
                with col4:
                    pending_alarms = len(alarm_df) - ack_alarms
                    st.metric("Pending Alarms", f"{pending_alarms:,}", delta=f"-{ack_alarms}")
                
                st.markdown("---")
                
                # Alarm timeline
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⏱️ Alarm Timeline")
                    alarm_timeline = alarm_df.groupby(
                        alarm_df['Event Time'].dt.floor('10min')
                    ).size().reset_index(name='count')
                    
                    fig_alarm_timeline = px.line(alarm_timeline, x='Event Time', y='count',
                                                 title="Alarm Frequency Over Time",
                                                 markers=True)
                    fig_alarm_timeline.update_traces(line_color='red')
                    st.plotly_chart(fig_alarm_timeline, width='stretch')
                
                with col2:
                    st.subheader("🎯 Critical Sources")
                    critical_sources_counts = alarm_df['Source'].value_counts().head(10)
                    fig_critical = px.bar(x=critical_sources_counts.values,
                                         y=critical_sources_counts.index,
                                         orientation='h',
                                         title="Top 10 Sources with Alarms",
                                         color=critical_sources_counts.values,
                                         color_continuous_scale='Reds')
                    fig_critical.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_critical, width='stretch')
                
                # Alarm types and acknowledgment
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔴 Alarm Types")
                    alarm_types = alarm_df['Condition'].value_counts().head(10)
                    fig_alarm_types = px.bar(x=alarm_types.index, y=alarm_types.values,
                                            title="Alarm Condition Distribution",
                                            color=alarm_types.values,
                                            color_continuous_scale='OrRd')
                    fig_alarm_types.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_alarm_types, width='stretch')
                
                with col2:
                    st.subheader("✅ Acknowledgment Status")
                    ack_status = alarm_df['Action'].apply(
                        lambda x: 'Acknowledged' if 'ACK' in str(x) else 'Not Acknowledged'
                    ).value_counts()
                    
                    fig_ack = px.pie(values=ack_status.values, names=ack_status.index,
                                    title="Alarm Acknowledgment Status",
                                    color_discrete_map={'Acknowledged': '#16A34A', 
                                                       'Not Acknowledged': '#F59E0B'})
                    st.plotly_chart(fig_ack, width='stretch')
            else:
                st.info("No alarms found in the selected data range")
        
        # Tab 3: Process Variables
        with tab3:
            st.header("📈 Process Variables Analysis")
            
            value_df = filtered_df[filtered_df['Value'].notna()]
            
            if len(value_df) > 0:
                # Statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Mean", f"{value_df['Value'].mean():.2f}")
                
                with col2:
                    st.metric("Median", f"{value_df['Value'].median():.2f}")
                
                with col3:
                    st.metric("Std Dev", f"{value_df['Value'].std():.2f}")
                
                with col4:
                    st.metric("Min", f"{value_df['Value'].min():.2f}")
                
                with col5:
                    st.metric("Max", f"{value_df['Value'].max():.2f}")
                
                st.markdown("---")
                
                # Value timeline
                st.subheader("📊 Values Over Time")
                fig_values = px.scatter(value_df, x='Event Time', y='Value',
                                       color='Source', hover_data=['Source', 'Condition', 'Units'],
                                       title="Process Values Timeline")
                fig_values.update_layout(height=400)
                st.plotly_chart(fig_values, width='stretch')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value distribution
                    st.subheader("📊 Value Distribution")
                    fig_hist = px.histogram(value_df, x='Value', nbins=50,
                                           title="Value Distribution Histogram")
                    fig_hist.add_vline(x=value_df['Value'].mean(), 
                                      line_dash="dash", line_color="#065F46",
                                      annotation_text="Mean")
                    fig_hist.add_vline(x=value_df['Value'].median(), 
                                      line_dash="dash", line_color="#10B981",
                                      annotation_text="Median")
                    st.plotly_chart(fig_hist, width='stretch')
                
                with col2:
                    # Box plot by units
                    st.subheader("📦 Values by Unit Type")
                    units_with_data = value_df[value_df['Units'] != '']['Units'].value_counts().head(5).index
                    value_df_units = value_df[value_df['Units'].isin(units_with_data)]
                    
                    if len(value_df_units) > 0:
                        fig_box = px.box(value_df_units, x='Units', y='Value',
                                        title="Value Ranges by Unit Type")
                        st.plotly_chart(fig_box, width='stretch')
                
                # Outlier detection
                st.subheader("🔍 Outlier Detection")
                
                Q1 = value_df['Value'].quantile(0.25)
                Q3 = value_df['Value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = value_df[(value_df['Value'] < lower_bound) | 
                                   (value_df['Value'] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Outliers Detected", f"{len(outliers):,}")
                with col2:
                    st.metric("Lower Bound", f"{lower_bound:.2f}")
                with col3:
                    st.metric("Upper Bound", f"{upper_bound:.2f}")
                
                # Outlier scatter plot
                fig_outliers = go.Figure()
                
                normal_data = value_df[(value_df['Value'] >= lower_bound) & 
                                       (value_df['Value'] <= upper_bound)]
                
                fig_outliers.add_trace(go.Scatter(
                    x=normal_data['Event Time'],
                    y=normal_data['Value'],
                    mode='markers',
                    name='Normal',
                    marker=dict(size=5, color='#16A34A', opacity=0.5)
                ))
                
                if len(outliers) > 0:
                    fig_outliers.add_trace(go.Scatter(
                        x=outliers['Event Time'],
                        y=outliers['Value'],
                        mode='markers',
                        name='Outliers',
                        marker=dict(size=10, color='red', symbol='triangle-up')
                    ))
                
                fig_outliers.add_hline(y=upper_bound, line_dash="dash", 
                                      line_color="#16A34A", opacity=0.5,
                                      annotation_text="Upper Bound")
                fig_outliers.add_hline(y=lower_bound, line_dash="dash", 
                                      line_color="#16A34A", opacity=0.5,
                                      annotation_text="Lower Bound")
                
                fig_outliers.update_layout(title="Outlier Detection (IQR Method)",
                                          xaxis_title="Time",
                                          yaxis_title="Value",
                                          height=400)
                
                st.plotly_chart(fig_outliers, width='stretch')
            else:
                st.info("No process values found in the selected data")
        
        # Tab 4: Data Table
        with tab4:
            st.header("📋 Event Data Table")
            
            # Search box
            search = st.text_input("🔍 Search in data", "")
            
            if search:
                mask = filtered_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
                display_df = filtered_df[mask]
            else:
                display_df = filtered_df
            
            # Show data info
            st.info(f"Showing {len(display_df):,} rows")
            
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_na = st.checkbox("Show rows with missing values only", False)
            with col2:
                show_recent = st.checkbox("Show most recent first", True)
            with col3:
                rows_to_show = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)
            
            if show_na:
                display_df = display_df[display_df.isna().any(axis=1)]
            
            if show_recent:
                display_df = display_df.sort_values('Event Time', ascending=False)
            
            # Pagination
            total_pages = len(display_df) // rows_to_show + (1 if len(display_df) % rows_to_show > 0 else 0)
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
            
            start_idx = (page - 1) * rows_to_show
            end_idx = min(start_idx + rows_to_show, len(display_df))
            
            # Display dataframe
            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                width='stretch',
                height=600
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_alarm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Tab 5: Summary Report
        with tab5:
            st.header("📑 Summary Report")
            
            # Generate report (Markdown)
            start_time = filtered_df['Event Time'].min()
            end_time = filtered_df['Event Time'].max()
            duration_hours = (end_time - start_time).total_seconds() / 3600 if end_time and start_time else 0
            avg_per_hour = (len(filtered_df) / duration_hours) if duration_hours > 0 else 0
            alarm_total = filtered_df['Is_Alarm'].sum()
            normal_total = len(filtered_df) - alarm_total
            ack_total = filtered_df['Action'].str.contains('ACK', na=False).sum()
            unack_total = len(filtered_df) - ack_total

            report_text = textwrap.dedent(f"""
            # ALARM DATA ANALYSIS REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            ## BASIC STATISTICS
            - Total Events: {len(filtered_df):,}
            - Time Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}
            - Duration: {duration_hours:.2f} hours
            - Average Events/Hour: {avg_per_hour:.2f}

            ## ALARM STATISTICS
            - Total Alarms: {alarm_total:,} ({(alarm_total/len(filtered_df))*100:.1f}%)
            - Normal Events: {normal_total:,} ({(normal_total/len(filtered_df))*100:.1f}%)

            ## ACKNOWLEDGMENT
            - Acknowledged Events: {ack_total:,}
            - Unacknowledged: {unack_total:,}

            ## TOP 5 EVENT SOURCES
            """)

            for i, (source, count) in enumerate(filtered_df['Source'].value_counts().head(5).items(), 1):
                report_text += f"\n{i}. {source}: {count:,} events ({count/len(filtered_df)*100:.1f}%)"

            report_text += "\n\n## TOP 5 CONDITIONS"
            for i, (condition, count) in enumerate(filtered_df['Condition'].value_counts().head(5).items(), 1):
                report_text += f"\n{i}. {condition}: {count:,} events ({count/len(filtered_df)*100:.1f}%)"

            report_text += "\n\n## TOP 5 LOCATIONS"
            for i, (location, count) in enumerate(filtered_df['Location Tag'].value_counts().head(5).items(), 1):
                report_text += f"\n{i}. {location}: {count:,} events ({count/len(filtered_df)*100:.1f}%)"

            # Value statistics
            value_data = filtered_df['Value'].dropna()
            if len(value_data) > 0:
                report_text += textwrap.dedent(f"""
                
                ## VALUE STATISTICS
                - Events with Values: {len(value_data):,} ({len(value_data)/len(filtered_df)*100:.1f}%)
                - Mean Value: {value_data.mean():.2f}
                - Median Value: {value_data.median():.2f}
                - Std Deviation: {value_data.std():.2f}
                - Min Value: {value_data.min():.2f}
                - Max Value: {value_data.max():.2f}
                """)

            # Display report as Markdown
            st.markdown(report_text)
            
            # Download report button
            st.download_button(
                label="📥 Download Report as Markdown",
                data=report_text,
                file_name=f"alarm_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"

            )
            
            # Additional insights
            st.markdown("---")
            st.subheader("🔍 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peak hour
                hourly_counts = filtered_df['Hour'].value_counts()
                peak_hour = hourly_counts.idxmax()
                st.info(f"**Peak Activity Hour:** {peak_hour:02d}:00 with {hourly_counts[peak_hour]:,} events")
                
                # Most problematic source
                top_source = filtered_df['Source'].value_counts().iloc[0]
                st.warning(f"**Most Active Source:** {filtered_df['Source'].value_counts().index[0]} ({top_source:,} events)")
            
            with col2:
                # Alarm rate
                alarm_rate = filtered_df['Is_Alarm'].sum() / len(filtered_df) * 100
                if alarm_rate > 20:
                    st.error(f"**High Alarm Rate:** {alarm_rate:.1f}% of events are alarms")
                else:
                    st.success(f"**Normal Alarm Rate:** {alarm_rate:.1f}% of events are alarms")
                
                # Acknowledgment rate
                ack_rate = filtered_df['Action'].str.contains('ACK', na=False).sum() / len(filtered_df) * 100
                if ack_rate < 50:
                    st.warning(f"**Low Acknowledgment Rate:** Only {ack_rate:.1f}% of events acknowledged")
                else:
                    st.success(f"**Good Acknowledgment Rate:** {ack_rate:.1f}% of events acknowledged")

        # Tab 6: Ask AI
        with tab6:
            st.header("🤖 Ask AI about your filtered data")
            st.caption("Only a compact summary (schema, small stats and a few sample rows) is sent to the AI.")

            # API key input (optional, for runtime convenience)
            with st.expander("API settings", expanded=False):
                current_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", "")
                key_input = st.text_input("OpenAI API Key", value=current_key, type="password")
                model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
                if st.button("Save API key"):
                    if key_input:
                        os.environ["OPENAI_API_KEY"] = key_input
                        st.session_state["OPENAI_API_KEY"] = key_input
                        st.success("API key saved for this session.")
                    else:
                        st.warning("No key entered.")

            mode = st.radio("Response type", ["Text answer", "Python code + run"], horizontal=True)
            question = st.text_area("Ask a question about the currently filtered data", placeholder="e.g., What are the top 3 sources by alarm count in this filtered range?")
            sample_rows = st.slider("Sample rows included in context", 3, 10, 5)
            col_run, col_show = st.columns([1,1])
            with col_run:
                run = st.button("Ask AI", type="primary")
            with col_show:
                show_ctx = st.checkbox("Show context sent to AI", value=False)

            if run and question:
                with st.spinner("Preparing context and querying AI..."):
                    context_text = build_ai_context(filtered_df, sample_rows=sample_rows)
                    code_mode = (mode == "Python code + run")
                    answer_text = ask_openai(question, context_text, model=model, code_mode=code_mode)

                if show_ctx:
                    with st.expander("Context sent to AI"):
                        st.code(context_text)

                if mode == "Text answer":
                    st.markdown("### Answer")
                    st.markdown(answer_text)
                else:
                    st.markdown("### Generated code")
                    code_block = extract_python_code(answer_text)
                    if not code_block:
                        st.error("The AI did not return a valid Python code block. Showing raw response below.")
                        st.code(answer_text)
                    else:
                        st.code(code_block, language="python")
                        with st.spinner("Executing code..."):
                            env, stdout_text, err = run_user_code(code_block, filtered_df.copy())
                        if err:
                            st.error("Execution failed. See traceback:")
                            st.code(err)
                        else:
                            # Show outputs
                            if 'fig' in env and env['fig'] is not None:
                                try:
                                    st.plotly_chart(env['fig'], width='stretch')
                                except Exception:
                                    st.warning("'fig' was not a valid Plotly figure.")
                            if 'result' in env:
                                res = env['result']
                                st.markdown("#### Result")
                                if isinstance(res, pd.DataFrame):
                                    st.dataframe(res, width='stretch', height=500)
                                elif isinstance(res, pd.Series):
                                    st.write(res)
                                else:
                                    st.write(res)
                            if stdout_text:
                                with st.expander("Console output"):
                                    st.code(stdout_text)
                            if ('result' not in env) and ('fig' not in env) and (not stdout_text):
                                st.info("Code ran but did not produce 'result', 'fig' or prints.")

else:
    # Welcome screen when no data is loaded
    st.info("👈 Please upload a CSV file using the sidebar to begin analysis")
    
    # Example data structure
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV file should contain the following columns:
    - **Event Time**: Timestamp of the event
    - **Location Tag**: Location identifier
    - **Source**: Source of the event
    - **Condition**: Event condition/type
    - **Action**: Action taken (ACK, etc.)
    - **Priority**: Event priority (optional)
    - **Description**: Event description
    - **Value**: Numerical value (optional)
    - **Units**: Measurement units (optional)
    """)
    
    # Sample data display
    sample_data = pd.DataFrame({
        'Event Time': ['2/1/2025 23:59:13', '2/1/2025 23:58:40'],
        'Location Tag': ['8800', '8800'],
        'Source': ['PIC8808', 'PIC8808'],
        'Condition': ['CHANGE', 'ALARM'],
        'Action': ['', 'ACK'],
        'Priority': [np.nan, np.nan],
        'Description': ['pida.op', 'DEAERATOR'],
        'Value': [20.0, 10.0],
        'Units': ['%', '%']
    })
    
    st.subheader("Sample Data")
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Alarm Analytics Dashboard v1.0 | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)