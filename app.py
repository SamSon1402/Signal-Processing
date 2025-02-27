import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
import time
import io
from PIL import Image
import torch
from modules.wave_extractor import WaveFeatureExtractor
from modules.wave_classifier import WaveClassifier, load_pretrained_model
from modules.visualizations import (
    generate_basic_visualizations,
    generate_3d_visualization,
    generate_feature_comparison
)
from modules.utils import apply_custom_css, card, st_matplotlib_figure, loading_animation

# Set page configuration
st.set_page_config(
    page_title="Wave Feature Extractor for Medical Imaging",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for futuristic UI
apply_custom_css()

# Initialize session state variables if they don't exist
if 'extractor' not in st.session_state:
    st.session_state.extractor = WaveFeatureExtractor(sampling_rate=1000)
if 'signal_generated' not in st.session_state:
    st.session_state.signal_generated = False
if 'current_tissue_type' not in st.session_state:
    st.session_state.current_tissue_type = "normal"
if 'current_abnormality' not in st.session_state:
    st.session_state.current_abnormality = 0.0
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = None
if 'extracted_features' not in st.session_state:
    st.session_state.extracted_features = None

# Function to generate signal
def generate_signal(tissue_type, abnormality_level):
    extractor = st.session_state.extractor
    t, x = extractor.generate_simulated_medical_wave(
        duration=1.0,
        tissue_type=tissue_type,
        abnormality_level=abnormality_level
    )
    st.session_state.current_signal = x
    st.session_state.current_time = t
    st.session_state.current_tissue_type = tissue_type
    st.session_state.current_abnormality = abnormality_level
    st.session_state.signal_generated = True
    return t, x

# Main app layout
def main():
    st.markdown('<h1 class="glow-text">Wave Feature Extractor for Medical Imaging</h1>', unsafe_allow_html=True)
    
    # Create sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
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
                generate_signal(tissue_type, abnormality_level)
        
        # Analysis options
        if st.session_state.signal_generated:
            st.markdown("### Analysis Options")
            
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
            
            if st.button("Extract Features", key="extract_button"):
                with st.spinner("Extracting features..."):
                    extractor = st.session_state.extractor
                    features = extractor.extract_all_features(
                        signal=st.session_state.current_signal, 
                        wavelet=wavelet_type, 
                        wavelet_level=wavelet_level
                    )
                    st.session_state.extracted_features = features
        
        # Model options
        st.markdown("### Classification")
        model_option = st.selectbox(
            "Model Type",
            ["CNN", "Random Forest", "SVM"],
            index=0
        )
        
        if st.session_state.signal_generated and st.button("Classify Signal", key="classify_button"):
            loading_animation()
            
            # Load and use the selected model
            model = load_pretrained_model(model_type=model_option)
            
            # Prepare input
            signal = st.session_state.current_signal
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(signal_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            st.session_state.classification_results = {
                "normal": float(probs[0]),
                "abnormal": float(probs[1]),
                "tumor": float(probs[2]),
                "cyst": float(probs[3])
            }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # If signal has been generated, show visualizations
        if st.session_state.signal_generated:
            tabs = st.tabs(["Time Domain", "Frequency Domain", "Time-Frequency", "Wavelet", "Features"])
            
            # Generate all visualizations
            visualizations = generate_basic_visualizations(
                st.session_state.extractor, 
                st.session_state.current_time, 
                st.session_state.current_signal
            )
            
            # Display appropriate visualization in each tab
            with tabs[0]:
                st.image(visualizations['time'], use_column_width=True)
                
            with tabs[1]:
                st.image(visualizations['fft'], use_column_width=True)
                
            with tabs[2]:
                st.plotly_chart(generate_3d_visualization(st.session_state.extractor), use_container_width=True)
                
            with tabs[3]:
                st.image(visualizations['wavelet'], use_column_width=True)
                
            with tabs[4]:
                if st.session_state.extracted_features is not None:
                    st.image(visualizations['features'], use_column_width=True)
                else:
                    st.info("Click 'Extract Features' in the sidebar to analyze the signal features.")
        else:
            # No signal generated yet
            st.info("Please generate a signal using the controls in the sidebar.")
            
            # Display example/demo content
            st.markdown("""
            ### About This Application
            
            This tool allows you to generate simulated wave signals similar to what would be produced
            in medical imaging scenarios. You can analyze these signals using various signal processing
            techniques:
            
            - **Time Domain Analysis**: View the raw signal waveform
            - **Frequency Domain Analysis**: Examine the frequency components using FFT
            - **Time-Frequency Analysis**: Visualize how frequency content changes over time
            - **Wavelet Analysis**: Perform multi-resolution analysis using wavelet transforms
            - **Feature Extraction**: Extract quantitative features from all domains
            
            ### Getting Started
            
            1. Select a tissue type from the sidebar
            2. Adjust the abnormality level (if applicable)
            3. Click "Generate Signal"
            4. Use the tabs above to explore different analyses
            5. Click "Extract Features" to perform detailed feature extraction
            """)
    
    with col2:
        # Current signal information
        if st.session_state.signal_generated:
            # Signal information card
            st.markdown(f"""
            <div class="card">
                <h3 class="glow-text">Signal Information</h3>
                <p><strong>Tissue Type:</strong> {st.session_state.current_tissue_type.capitalize()}</p>
                <p><strong>Abnormality Level:</strong> {st.session_state.current_abnormality:.2f}</p>
                <p><strong>Sampling Rate:</strong> {st.session_state.extractor.sampling_rate} Hz</p>
                <p><strong>Signal Length:</strong> {len(st.session_state.current_signal)} samples</p>
                <p><strong>Duration:</strong> {len(st.session_state.current_signal)/st.session_state.extractor.sampling_rate:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification results
            if 'classification_results' in st.session_state:
                st.markdown('<div class="card"><h3 class="glow-text">Classification Results</h3>', unsafe_allow_html=True)
                
                # Create a bar chart for the classification probabilities
                results = st.session_state.classification_results
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(results.values()),
                        y=list(results.keys()),
                        orientation='h',
                        marker=dict(
                            color=['rgba(76, 131, 255, 0.8)', 
                                   'rgba(0, 242, 255, 0.8)', 
                                   'rgba(255, 100, 100, 0.8)', 
                                   'rgba(144, 238, 144, 0.8)'],
                            line=dict(color='rgba(50, 50, 50, 0.8)', width=1)
                        )
                    )
                ])
                
                fig.update_layout(
                    title="Tissue Type Probability",
                    xaxis_title="Probability",
                    yaxis_title="Type",
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#ffffff"),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the predicted class
                predicted_class = max(results, key=results.get)
                confidence = results[predicted_class] * 100
                
                st.markdown(f"""
                <p style="text-align:center; font-size:1.2em;">
                    Predicted class: <span style="color:#00f2ff; font-weight:bold;">{predicted_class.upper()}</span><br>
                    Confidence: <span style="color:#4c83ff; font-weight:bold;">{confidence:.1f}%</span>
                </p>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature values card (if features have been extracted)
            if st.session_state.extracted_features is not None:
                with st.expander("Show Extracted Features"):
                    # Convert feature dictionary to DataFrame for display
                    features_df = pd.DataFrame({
                        'Feature': list(st.session_state.extracted_features.keys()),
                        'Value': list(st.session_state.extracted_features.values())
                    })
                    
                    # Display the features table
                    st.dataframe(features_df, use_container_width=True)
        
        # Help and information
        with st.expander("Help & Information"):
            st.markdown("""
            ### Signal Type Information
            
            - **Normal**: Represents typical tissue response with low noise and predictable reflections
            - **Abnormal**: Shows slight deviations from normal patterns with medium amplitude reflections
            - **Tumor**: Exhibits strong reflections at specific frequency ranges and higher amplitude patterns
            - **Cyst**: Features characteristic periodic patterns in the signal
            
            ### Analysis Methods
            
            - **FFT**: Fast Fourier Transform reveals frequency components
            - **STFT**: Short-Time Fourier Transform shows how frequencies change over time
            - **Wavelet**: Multi-resolution analysis that preserves both time and frequency information
            
            ### Feature Extraction
            
            Features are extracted from three domains:
            - Time domain (statistical properties of the raw signal)
            - Frequency domain (spectral characteristics)
            - Wavelet domain (multi-resolution information)
            """)

if __name__ == "__main__":
    main()