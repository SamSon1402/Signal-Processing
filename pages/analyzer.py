import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

from modules.wave_extractor import WaveFeatureExtractor
from modules.visualizations import (
    generate_basic_visualizations,
    generate_3d_visualization,
    generate_feature_comparison
)
from modules.utils import (
    apply_custom_css,
    card,
    st_matplotlib_figure,
    loading_animation,
    create_feature_table,
    export_features_to_csv,
    generate_download_link
)

# Set page configuration
st.set_page_config(
    page_title="Wave Feature Extractor - Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state variables if they don't exist
if 'extractor' not in st.session_state:
    st.session_state.extractor = WaveFeatureExtractor(sampling_rate=1000)
if 'analyzer_signal_generated' not in st.session_state:
    st.session_state.analyzer_signal_generated = False
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = None
if 'extracted_features' not in st.session_state:
    st.session_state.extracted_features = None

def main():
    # Page title
    st.markdown('<h1 class="glow-text">Signal Analyzer</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="card">
        <p>
            The Signal Analyzer provides detailed analysis of wave signals in multiple domains:
            time domain, frequency domain, time-frequency representation, and wavelet analysis.
            You can extract and compare features from these domains to identify patterns and characteristics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## Analysis Controls")
        
        # Signal selection section
        st.markdown("### Signal Source")
        signal_source = st.radio(
            "Select signal source",
            ["Generate New", "Use Existing", "Upload Signal"],
            index=0
        )
        
        if signal_source == "Generate New":
            # Signal generation parameters
            st.markdown("### Signal Parameters")
            tissue_type = st.selectbox(
                "Tissue Type", 
                ["normal", "abnormal", "tumor", "cyst"],
                index=0
            )
            
            abnormality_level = st.slider(
                "Abnormality Level", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5 if tissue_type != "normal" else 0.0,
                step=0.1,
                disabled=(tissue_type=="normal")
            )
            
            sampling_rate = st.slider(
                "Sampling Rate (Hz)", 
                min_value=500, 
                max_value=2000, 
                value=1000,
                step=100
            )
            
            # Update the extractor with new sampling rate if changed
            if sampling_rate != st.session_state.extractor.sampling_rate:
                st.session_state.extractor = WaveFeatureExtractor(sampling_rate=sampling_rate)
            
            # Generate signal button
            if st.button("Generate Signal", key="generate_button"):
                with st.spinner("Generating signal..."):
                    extractor = st.session_state.extractor
                    t, x = extractor.generate_simulated_medical_wave(
                        duration=1.0,
                        tissue_type=tissue_type,
                        abnormality_level=abnormality_level
                    )
                    st.session_state.current_signal = x
                    st.session_state.current_time = t
                    st.session_state.analyzer_signal_generated = True
                    st.session_state.current_tissue_type = tissue_type
                    st.session_state.current_abnormality = abnormality_level
                    
                    # Reset extracted features
                    if 'extracted_features' in st.session_state:
                        del st.session_state.extracted_features
        
        elif signal_source == "Use Existing":
            # Use existing signal from session state
            if 'simulator_signals' in st.session_state and st.session_state.simulator_signals:
                signal_options = list(st.session_state.simulator_signals.keys())
                selected_signal = st.selectbox(
                    "Select saved signal",
                    signal_options,
                    index=0
                )
                
                if st.button("Load Signal", key="load_existing_button"):
                    signal_data = st.session_state.simulator_signals[selected_signal]
                    
                    # Update extractor with selected signal
                    extractor = WaveFeatureExtractor(sampling_rate=signal_data['sampling_rate'])
                    t = signal_data['time']
                    x = signal_data['signal']
                    
                    # Store in session state
                    st.session_state.extractor = extractor
                    st.session_state.current_signal = x
                    st.session_state.current_time = t
                    st.session_state.analyzer_signal_generated = True
                    st.session_state.current_tissue_type = signal_data.get('tissue_type', 'unknown')
                    st.session_state.current_abnormality = signal_data.get('abnormality_level', 0.0)
                    
                    # Reset extracted features
                    if 'extracted_features' in st.session_state:
                        del st.session_state.extracted_features
            else:
                st.info("No saved signals found. Generate signals in the Wave Simulator first.")
        
        elif signal_source == "Upload Signal":
            # File uploader for signals
            uploaded_file = st.file_uploader("Upload signal data (CSV)", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    df = pd.read_csv(uploaded_file)
                    
                    # Check for required columns
                    if 'time' in df.columns and 'amplitude' in df.columns:
                        sampling_rate = st.number_input(
                            "Sampling Rate (Hz)",
                            min_value=100,
                            max_value=10000,
                            value=1000,
                            step=100
                        )
                        
                        if st.button("Process Signal", key="process_upload_button"):
                            # Extract time and amplitude columns
                            t = df['time'].values
                            x = df['amplitude'].values
                            
                            # Update extractor
                            extractor = WaveFeatureExtractor(sampling_rate=sampling_rate)
                            
                            # Store in session state
                            st.session_state.extractor = extractor
                            st.session_state.current_signal = x
                            st.session_state.current_time = t
                            st.session_state.analyzer_signal_generated = True
                            st.session_state.current_tissue_type = "uploaded"
                            st.session_state.current_abnormality = 0.0
                            
                            # Reset extracted features
                            if 'extracted_features' in st.session_state:
                                del st.session_state.extracted_features
                    else:
                        st.error("CSV file must contain 'time' and 'amplitude' columns.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # Analysis options
        if st.session_state.analyzer_signal_generated:
            st.markdown("### Analysis Options")
            
            analysis_type = st.radio(
                "Analysis Type",
                ["Basic Analysis", "Feature Extraction", "3D Visualization"],
                index=0
            )
            
            if analysis_type == "Feature Extraction":
                wavelet_type = st.selectbox(
                    "Wavelet Type",
                    ["db4", "sym5", "coif3", "haar"],
                    index=0
                )
                
                wavelet_level = st.slider(
                    "Decomposition Level",
                    min_value=2,
                    max_value=8,
                    value=5,
                    step=1
                )
                
                feature_domains = st.multiselect(
                    "Feature Domains",
                    ["Time Domain", "Frequency Domain", "Wavelet Domain"],
                    default=["Time Domain", "Frequency Domain", "Wavelet Domain"]
                )
                
                if st.button("Extract Features", key="extract_button"):
                    with st.spinner("Extracting features..."):
                        extractor = st.session_state.extractor
                        features = extractor.extract_all_features(
                            signal=st.session_state.current_signal, 
                            wavelet=wavelet_type, 
                            wavelet_level=wavelet_level
                        )
                        st.session_state.extracted_features = features
                        st.session_state.feature_params = {
                            'wavelet_type': wavelet_type,
                            'wavelet_level': wavelet_level,
                            'domains': feature_domains
                        }
                        
                if 'extracted_features' in st.session_state and st.session_state.extracted_features:
                    export_csv = export_features_to_csv(st.session_state.extracted_features)
                    generate_download_link(export_csv, "signal_features.csv", "Download Features as CSV")
    
    # Main content area
    if st.session_state.analyzer_signal_generated:
        # Get the current signal
        extractor = st.session_state.extractor
        
        # Create tabs for different analysis views
        tabs = st.tabs(["Time Domain", "Frequency Domain", "Time-Frequency", "Wavelet", "Features"])
        
        # Generate all visualizations
        visualizations = generate_basic_visualizations(
            extractor, 
            st.session_state.current_time, 
            st.session_state.current_signal
        )
        
        # Display appropriate visualization in each tab
        with tabs[0]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(visualizations['time'], use_column_width=True)
            
            with col2:
                # Signal information card
                st.markdown(f"""
                <div class="card">
                    <h3 class="glow-text">Time Domain Analysis</h3>
                    <p>Time domain analysis examines the raw signal as it varies over time, showing amplitude fluctuations directly.</p>
                    <p><strong>Signal Length:</strong> {len(st.session_state.current_signal)} samples</p>
                    <p><strong>Duration:</strong> {len(st.session_state.current_signal)/st.session_state.extractor.sampling_rate:.2f} seconds</p>
                    <p><strong>Sampling Rate:</strong> {st.session_state.extractor.sampling_rate} Hz</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Time domain metrics
                if 'time_features' in extractor.results:
                    features = extractor.results['time_features']
                    st.markdown(f"""
                    <div class="card">
                        <h3 class="glow-text">Signal Statistics</h3>
                        <p><strong>Mean:</strong> {features['mean']:.4f}</p>
                        <p><strong>Std Dev:</strong> {features['std']:.4f}</p>
                        <p><strong>Min:</strong> {features['min']:.4f}</p>
                        <p><strong>Max:</strong> {features['max']:.4f}</p>
                        <p><strong>RMS:</strong> {features['rms']:.4f}</p>
                        <p><strong>Zero Crossings:</strong> {features['zero_crossing_rate']*len(st.session_state.current_signal):.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        with tabs[1]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(visualizations['fft'], use_column_width=True)
            
            with col2:
                # Frequency domain information
                st.markdown(f"""
                <div class="card">
                    <h3 class="glow-text">Frequency Domain Analysis</h3>
                    <p>Frequency domain analysis reveals the component frequencies present in the signal using Fast Fourier Transform (FFT).</p>
                    <p><strong>Frequency Resolution:</strong> {st.session_state.extractor.sampling_rate/len(st.session_state.current_signal):.2f} Hz</p>
                    <p><strong>Max Frequency:</strong> {st.session_state.extractor.sampling_rate/2} Hz (Nyquist limit)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Frequency domain metrics
                if 'frequency_features' in extractor.results:
                    features = extractor.results['frequency_features']
                    st.markdown(f"""
                    <div class="card">
                        <h3 class="glow-text">Spectral Statistics</h3>
                        <p><strong>Spectral Centroid:</strong> {features['spectral_centroid']:.2f} Hz</p>
                        <p><strong>Spectral Bandwidth:</strong> {features['spectral_bandwidth']:.2f} Hz</p>
                        <p><strong>Spectral Rolloff:</strong> {features['spectral_rolloff']:.2f} Hz</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display dominant frequency peaks
                    st.markdown('<div class="card"><h3 class="glow-text">Dominant Frequencies</h3>', unsafe_allow_html=True)
                    
                    # Find all peak features
                    peak_features = {k: v for k, v in features.items() if 'peak_' in k and 'freq' in k}
                    
                    # Create a table of peaks
                    peak_data = []
                    for i in range(1, 6):  # Assuming up to 5 peaks
                        freq_key = f'peak_{i}_freq'
                        mag_key = f'peak_{i}_magnitude'
                        if freq_key in features and mag_key in features:
                            peak_data.append({
                                'Peak': i,
                                'Frequency (Hz)': f"{features[freq_key]:.2f}",
                                'Magnitude': f"{features[mag_key]:.2f}"
                            })
                    
                    # Display the peak data
                    peak_df = pd.DataFrame(peak_data)
                    st.table(peak_df)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[2]:
            # 3D time-frequency visualization
            st.plotly_chart(generate_3d_visualization(extractor), use_container_width=True)
            
            # Time-frequency information
            st.markdown(f"""
            <div class="card">
                <h3 class="glow-text">Time-Frequency Analysis</h3>
                <p>
                    Time-frequency analysis shows how the frequency content of the signal changes over time. 
                    This is achieved using the Short-Time Fourier Transform (STFT), which applies FFT to small, 
                    overlapping windows of the signal.
                </p>
                <p>
                    The 3D visualization represents:
                </p>
                <ul>
                    <li><strong>X-axis:</strong> Time (seconds)</li>
                    <li><strong>Y-axis:</strong> Frequency (Hz)</li>
                    <li><strong>Z-axis/Color:</strong> Magnitude (intensity)</li>
                </ul>
                <p>
                    Patterns and features in this visualization can reveal:
                </p>
                <ul>
                    <li>Transient events and their frequency characteristics</li>
                    <li>Changes in frequency content over time</li>
                    <li>Distinctive patterns specific to different tissue types</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with tabs[3]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(visualizations['wavelet'], use_column_width=True)
            
            with col2:
                # Wavelet analysis information
                st.markdown(f"""
                <div class="card">
                    <h3 class="glow-text">Wavelet Analysis</h3>
                    <p>
                        Wavelet analysis provides multi-resolution decomposition of the signal, enabling the examination 
                        of both time and frequency characteristics simultaneously at different scales.
                    </p>
                    <p>
                        The signal is decomposed into:
                    </p>
                    <ul>
                        <li><strong>Approximation Coefficients:</strong> Low-frequency components</li>
                        <li><strong>Detail Coefficients:</strong> High-frequency components at different scales</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Wavelet features if available
                if 'wavelet_features' in extractor.results:
                    features = extractor.results['wavelet_features']
                    
                    # Wavelet energy distribution
                    energy_features = {k: v for k, v in features.items() if 'energy' in k}
                    total_energy = sum(energy_features.values())
                    energy_distribution = {k: v/total_energy*100 for k, v in energy_features.items()}
                    
                    st.markdown('<div class="card"><h3 class="glow-text">Wavelet Energy Distribution</h3>', unsafe_allow_html=True)
                    
                    # Create a small bar chart for energy distribution
                    fig = plt.figure(figsize=(8, 4), facecolor='#0e1117')
                    ax = fig.add_subplot(111)
                    
                    labels = list(energy_distribution.keys())
                    values = list(energy_distribution.values())
                    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
                    
                    ax.bar(labels, values, color=colors)
                    ax.set_ylabel('Energy %', color='white')
                    ax.set_title('Wavelet Coefficient Energy Distribution', color='white')
                    ax.tick_params(colors='white')
                    plt.xticks(rotation=45, ha='right')
                    ax.set_facecolor('#0e1117')
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[4]:
            if 'extracted_features' in st.session_state and st.session_state.extracted_features:
                features = st.session_state.extracted_features
                
                # Feature visualization
                st.image(visualizations['features'], use_column_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Time domain summary
                    time_features = {k: v for k, v in features.items() if k.startswith('time_')}
                    st.markdown('<div class="card"><h3 class="glow-text">Time Domain Features</h3>', unsafe_allow_html=True)
                    create_feature_table(time_features, top_n=10, key_prefix="time")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Frequency domain summary
                    freq_features = {k: v for k, v in features.items() if k.startswith('freq_')}
                    st.markdown('<div class="card"><h3 class="glow-text">Frequency Domain Features</h3>', unsafe_allow_html=True)
                    create_feature_table(freq_features, top_n=10, key_prefix="freq")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Wavelet domain summary
                    wavelet_features = {k: v for k, v in features.items() if k.startswith('wavelet_')}
                    st.markdown('<div class="card"><h3 class="glow-text">Wavelet Domain Features</h3>', unsafe_allow_html=True)
                    create_feature_table(wavelet_features, top_n=10, key_prefix="wavelet")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature analysis summary
                st.markdown('<div class="card"><h3 class="glow-text">Feature Analysis Summary</h3>', unsafe_allow_html=True)
                
                # Create two columns for the summary
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    # Summary information
                    st.markdown("""
                    <h4>Most Significant Features</h4>
                    <p>
                        The most significant features often provide the strongest discriminative power for 
                        classification and analysis of different tissue types. The top features shown in the 
                        visualization above highlight the characteristics that distinguish this signal.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Display signal parameters
                    st.markdown("<h4>Signal Parameters</h4>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <ul>
                        <li><strong>Tissue Type:</strong> {st.session_state.current_tissue_type.capitalize()}</li>
                        <li><strong>Abnormality Level:</strong> {st.session_state.current_abnormality:.2f}</li>
                        <li><strong>Sampling Rate:</strong> {st.session_state.extractor.sampling_rate} Hz</li>
                        <li><strong>Duration:</strong> {len(st.session_state.current_signal)/st.session_state.extractor.sampling_rate:.2f} seconds</li>
                        <li><strong>Wavelet Type:</strong> {st.session_state.feature_params.get('wavelet_type', 'N/A')}</li>
                        <li><strong>Wavelet Level:</strong> {st.session_state.feature_params.get('wavelet_level', 'N/A')}</li>
                    </ul>
                    """, unsafe_allow_html=True)
                
                with summary_col2:
                    # Domain contribution
                    st.markdown("<h4>Domain Contribution Analysis</h4>", unsafe_allow_html=True)
                    
                    # Calculate feature variance by domain
                    time_var = np.var(list(time_features.values()))
                    freq_var = np.var(list(freq_features.values()))
                    wavelet_var = np.var(list(wavelet_features.values()))
                    
                    total_var = time_var + freq_var + wavelet_var
                    
                    # Create a pie chart for domain contribution
                    fig = plt.figure(figsize=(8, 5), facecolor='#0e1117')
                    ax = fig.add_subplot(111)
                    
                    labels = ['Time Domain', 'Frequency Domain', 'Wavelet Domain']
                    sizes = [time_var/total_var*100, freq_var/total_var*100, wavelet_var/total_var*100]
                    colors = ['#4c83ff', '#00f2ff', '#9c6eff']
                    
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                          shadow=False, startangle=90)
                    ax.axis('equal')
                    ax.set_title('Feature Variance by Domain', color='white')
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Domain summary
                    st.markdown(f"""
                    <p>
                        This chart shows the contribution of each domain to the overall feature variance.
                        Domains with higher variance typically contain more discriminative features for classification.
                    </p>
                    <p>
                        <strong>Total Features Extracted:</strong> {len(features)}
                    </p>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    export_csv = export_features_to_csv(features)
                    generate_download_link(export_csv, "all_features.csv", "Download All Features (CSV)")
                
                with col2:
                    # Export as JSON option
                    import json
                    json_str = json.dumps(features, indent=2)
                    generate_download_link(json_str, "features.json", "Download as JSON", mime="application/json")
            else:
                # No features extracted yet
                st.info("Click 'Extract Features' in the sidebar to analyze the signal and extract features.")
    else:
        # No signal generated yet
        st.info("Please generate or upload a signal using the controls in the sidebar.")

if __name__ == "__main__":
    main()