import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import time

def apply_custom_css():
    """Apply custom CSS styling for a futuristic UI."""
    st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        .stApp {
            background-image: linear-gradient(to bottom right, #0e1117, #1a1c24);
        }
        
        h1, h2, h3 {
            color: #00c3ff;
            font-family: 'Arial', sans-serif;
        }
        
        .stButton>button {
            background-color: #2e4b7c;
            color: white;
            border-radius: 20px;
            border: 1px solid #4c83ff;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #4c83ff;
            border: 1px solid #2e4b7c;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 131, 255, 0.4);
        }
        
        .css-1kyxreq, .css-12oz5g7 {
            background-color: rgba(14, 17, 23, 0.7);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #00f2ff;
        }
        
        .css-1oe6wy4 {
            background-color: #1f2937;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #4c83ff;
        }
        
        /* Sidebar */
        .css-6qob1r {
            background-image: linear-gradient(to bottom, #111827, #1f2937);
        }
        
        /* Notification and banner colors */
        div[data-baseweb="notification"] {
            background-color: #2e4b7c;
            border-radius: 10px;
        }
        
        div[data-testid="stMarkdownContainer"] code {
            background-color: #1a1c24;
            color: #00f2ff;
            padding: 4px 8px;
            border-radius: 5px;
        }
        
        /* Card-like container */
        .card {
            background-color: rgba(31, 41, 55, 0.7);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* Glow effect for headers */
        .glow-text {
            color: #4c83ff;
            text-shadow: 0 0 10px rgba(76, 131, 255, 0.7);
        }
        
        /* Table styling */
        .dataframe {
            background-color: rgba(31, 41, 55, 0.5);
            border-radius: 10px;
        }
        
        .dataframe th {
            background-color: rgba(76, 131, 255, 0.2);
            color: #ffffff;
            padding: 8px;
        }
        
        .dataframe td {
            color: #ffffff;
            padding: 8px;
        }
        
        /* Slider styling */
        div[data-testid="stSlider"] > div {
            background-color: rgba(76, 131, 255, 0.2);
        }
        
        /* Selectbox styling */
        div[data-baseweb="select"] > div {
            background-color: rgba(31, 41, 55, 0.7);
            border-color: #4c83ff;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(31, 41, 55, 0.7);
            border-radius: 10px;
            color: #ffffff;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(17, 24, 39, 0.7);
            border-radius: 10px;
            padding: 5px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            color: #ffffff;
            background-color: rgba(31, 41, 55, 0.7);
            border: 1px solid rgba(76, 131, 255, 0.3);
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(76, 131, 255, 0.4);
            border: 1px solid #4c83ff;
        }
        
        /* Footer styling */
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

def card(title, content):
    """
    Create a card-like container for content.
    
    Parameters:
    -----------
    title : str
        Card title
    content : str
        Card content in HTML format
        
    Returns:
    --------
    None
    """
    st.markdown(f"""
    <div class="card">
        <h3 class="glow-text">{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

def st_matplotlib_figure(fig):
    """
    Convert a matplotlib figure to an image for display in Streamlit.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to convert
        
    Returns:
    --------
    image : PIL.Image
        Converted image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', 
                facecolor='#0e1117', edgecolor='none')
    buf.seek(0)
    return Image.open(buf)

def loading_animation():
    """
    Display an animated loading progress bar.
    
    Returns:
    --------
    None
    """
    progress_text = "Processing signal data... Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

def create_metric_card(label, value, delta=None, help_text=None):
    """
    Create a styled metric card.
    
    Parameters:
    -----------
    label : str
        Metric label
    value : str or float
        Metric value
    delta : str or float, optional
        Delta value to show change
    help_text : str, optional
        Tooltip help text
        
    Returns:
    --------
    None
    """
    if delta is not None:
        st.metric(label=label, value=value, delta=delta, help=help_text)
    else:
        st.metric(label=label, value=value, help=help_text)

def create_feature_table(features, top_n=10, key_prefix=""):
    """
    Create a styled table for displaying features.
    
    Parameters:
    -----------
    features : dict
        Dictionary of feature names and values
    top_n : int, optional
        Number of top features to display
    key_prefix : str, optional
        Prefix for unique keys
        
    Returns:
    --------
    None
    """
    # Convert to pandas Series and sort by absolute value
    import pandas as pd
    features_series = pd.Series(features)
    sorted_features = features_series.abs().sort_values(ascending=False)
    
    # Create DataFrame with top features
    top_features = sorted_features.head(top_n)
    df = pd.DataFrame({
        'Feature': top_features.index,
        'Value': [features[f] for f in top_features.index]
    })
    
    # Display as a styled table
    st.dataframe(df, use_container_width=True)

def format_signal_parameters(params):
    """
    Format signal parameters for display.
    
    Parameters:
    -----------
    params : dict
        Dictionary of signal parameters
        
    Returns:
    --------
    html : str
        Formatted HTML string
    """
    html = "<ul style='list-style-type:none; padding-left:0;'>"
    
    for key, value in params.items():
        if key == 'frequencies' and isinstance(value, list):
            html += f"<li><strong>{key.capitalize()}:</strong> {', '.join([str(v) for v in value])}</li>"
        else:
            html += f"<li><strong>{key.capitalize()}:</strong> {value}</li>"
    
    html += "</ul>"
    return html

def display_help_tooltip(text, icon="ℹ️"):
    """
    Display a help tooltip with an icon.
    
    Parameters:
    -----------
    text : str
        Help text
    icon : str, optional
        Icon to display
        
    Returns:
    --------
    None
    """
    st.markdown(f"""
    <div style="position: relative; display: inline-block; margin-left: 5px;">
        <span style="cursor: pointer; font-size: 16px;">{icon}</span>
        <div style="visibility: hidden; width: 250px; background-color: rgba(31, 41, 55, 0.9); 
                    color: #fff; text-align: left; border-radius: 6px; padding: 8px; 
                    position: absolute; z-index: 1; bottom: 125%; left: 50%; 
                    margin-left: -125px; opacity: 0; transition: opacity 0.3s; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2); border: 1px solid #4c83ff;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def visualize_signal_comparison(signal1, signal2, labels=None, sampling_rate=1000):
    """
    Create a visualization comparing two signals.
    
    Parameters:
    -----------
    signal1 : numpy.ndarray
        First signal
    signal2 : numpy.ndarray
        Second signal
    labels : list, optional
        Labels for the signals
    sampling_rate : int, optional
        Sampling rate in Hz
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The comparison figure
    """
    if labels is None:
        labels = ['Signal 1', 'Signal 2']
    
    # Create time arrays
    t1 = np.arange(len(signal1)) / sampling_rate
    t2 = np.arange(len(signal2)) / sampling_rate
    
    # Create figure
    fig = plt.figure(figsize=(10, 6), facecolor='#0e1117')
    ax = fig.add_subplot(111)
    
    # Plot signals
    ax.plot(t1, signal1, label=labels[0], color='#4c83ff', linewidth=1.5)
    ax.plot(t2, signal2, label=labels[1], color='#00f2ff', linewidth=1.5)
    
    # Add labels and title
    ax.set_title('Signal Comparison', color='white', fontsize=14)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1c24', edgecolor='none', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set background color
    ax.set_facecolor('#0e1117')
    
    fig.tight_layout()
    return fig

def generate_download_link(data, filename, label="Download Data", mime="text/csv"):
    """
    Generate a download link for data.
    
    Parameters:
    -----------
    data : bytes or str
        Data to download
    filename : str
        Filename for download
    label : str, optional
        Link label
    mime : str, optional
        MIME type
        
    Returns:
    --------
    link : str
        HTML download link
    """
    import base64
    
    if isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
    else:
        b64 = base64.b64encode(data).decode()
    
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}" class="download-link">{label}</a>'
    
    st.markdown(f"""
    <style>
    .download-link {{
        display: inline-block;
        background-color: #2e4b7c;
        color: white;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        border-radius: 20px;
        border: 1px solid #4c83ff;
        transition: all 0.3s ease;
    }}
    .download-link:hover {{
        background-color: #4c83ff;
        border: 1px solid #2e4b7c;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 131, 255, 0.4);
    }}
    </style>
    {href}
    """, unsafe_allow_html=True)

def export_features_to_csv(features):
    """
    Export features to CSV format.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features
        
    Returns:
    --------
    csv_data : str
        CSV formatted data
    """
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Feature', 'Value'])
    
    # Write data
    for key, value in features.items():
        writer.writerow([key, value])
    
    return output.getvalue()

def create_directory_structure():
    """
    Create the necessary directory structure for the application.
    
    Returns:
    --------
    None
    """
    import os
    
    # Create directories
    directories = [
        "models",
        "models/pretrained",
        "assets",
        "assets/css",
        "assets/images",
        "assets/sample_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)