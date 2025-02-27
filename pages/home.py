import streamlit as st
import plotly.graph_objects as go
import numpy as np
from modules.utils import apply_custom_css, card

# Set page configuration
st.set_page_config(
    page_title="Wave Feature Extractor - Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

def main():
    # Page title
    st.markdown('<h1 class="glow-text">Wave Feature Extractor for Medical Imaging</h1>', unsafe_allow_html=True)
    
    # Hero section with visualization preview
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 class="glow-text">Advanced Signal Processing Tool</h2>
            <p style="font-size: 1.2em; margin-bottom: 20px;">
                Analyze, extract features, and classify wave signals for medical imaging applications
            </p>
            <p>
                This powerful tool demonstrates advanced signal processing techniques for analyzing 
                wave-based data patterns commonly found in medical imaging scenarios. It combines 
                time-domain, frequency-domain, and wavelet analysis with machine learning to provide 
                comprehensive signal insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown('<h3 class="glow-text">Key Features</h3>', unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            card("Signal Analysis", """
            <ul>
                <li>Time-Domain Analysis</li>
                <li>Frequency-Domain Analysis</li>
                <li>Wavelet Transformation</li>
                <li>Time-Frequency Analysis</li>
            </ul>
            """)
            
        with col_b:
            card("Feature Extraction", """
            <ul>
                <li>Statistical Features</li>
                <li>Spectral Features</li>
                <li>Wavelet-based Features</li>
                <li>Pattern Recognition</li>
            </ul>
            """)
            
        with col_c:
            card("Classification", """
            <ul>
                <li>Neural Network Models</li>
                <li>Random Forest</li>
                <li>SVM Classification</li>
                <li>Tissue Type Detection</li>
            </ul>
            """)
    
    with col2:
        # Generate a sample 3D visualization for the homepage
        # Create sample data for the visualization
        t = np.linspace(0, 1, 50)
        f = np.linspace(1, 50, 50)
        
        # Create a sample spectrogram
        Zxx = np.zeros((len(f), len(t)))
        for i, freq in enumerate(f):
            for j, time in enumerate(t):
                Zxx[i, j] = np.sin(2 * np.pi * freq * time) * np.exp(-((time-0.5)**2)/0.1) * np.exp(-((freq-25)**2)/200)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=np.abs(Zxx),
            x=t,
            y=f,
            colorscale='Viridis',
            surfacecolor=np.abs(Zxx),
            colorbar=dict(title='Magnitude')
        )])
        
        fig.update_layout(
            title='Signal Analysis in 3D',
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                zaxis_title='Magnitude',
                xaxis=dict(showbackground=False, gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(showbackground=False, gridcolor='rgba(255, 255, 255, 0.1)'),
                zaxis=dict(showbackground=False, gridcolor='rgba(255, 255, 255, 0.1)'),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                )
            ),
            width=600,
            height=500,
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='rgba(0,0,0,0)',
            scene_bgcolor='#111827',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick links to other pages
        st.markdown("""
        <div class="card">
            <h3 class="glow-text">Quick Navigation</h3>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <a href="/analyzer" target="_self" style="text-decoration: none;">
                    <div style="background-color: rgba(76, 131, 255, 0.3); border: 1px solid #4c83ff; border-radius: 10px; padding: 10px; transition: all 0.3s ease;">
                        <h4 style="margin: 0; color: white;">Signal Analyzer</h4>
                        <p style="margin: 5px 0 0 0; color: #ccc; font-size: 0.9em;">Analyze wave signals in multiple domains</p>
                    </div>
                </a>
                <a href="/simulator" target="_self" style="text-decoration: none;">
                    <div style="background-color: rgba(76, 131, 255, 0.3); border: 1px solid #4c83ff; border-radius: 10px; padding: 10px; transition: all 0.3s ease;">
                        <h4 style="margin: 0; color: white;">Wave Simulator</h4>
                        <p style="margin: 5px 0 0 0; color: #ccc; font-size: 0.9em;">Generate simulated medical wave signals</p>
                    </div>
                </a>
                <a href="/classifier" target="_self" style="text-decoration: none;">
                    <div style="background-color: rgba(76, 131, 255, 0.3); border: 1px solid #4c83ff; border-radius: 10px; padding: 10px; transition: all 0.3s ease;">
                        <h4 style="margin: 0; color: white;">Signal Classifier</h4>
                        <p style="margin: 5px 0 0 0; color: #ccc; font-size: 0.9em;">Classify tissue types using AI models</p>
                    </div>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Application description
    st.markdown('<h3 class="glow-text">About This Application</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>
            The <strong>Wave Feature Extractor for Medical Imaging</strong> is an advanced signal processing application 
            designed to demonstrate how different tissue types in medical imaging scenarios can be analyzed and classified 
            using sophisticated signal processing and machine learning techniques.
        </p>
        
        <p>
            This application focuses on extracting meaningful features from wave-based data patterns through multiple 
            analysis domains:
        </p>
        
        <ul>
            <li><strong>Time Domain Analysis:</strong> Examine raw signal characteristics over time</li>
            <li><strong>Frequency Domain Analysis:</strong> Identify frequency components using Fast Fourier Transform (FFT)</li>
            <li><strong>Time-Frequency Analysis:</strong> Visualize how frequencies change over time with Short-Time Fourier Transform (STFT)</li>
            <li><strong>Wavelet Analysis:</strong> Perform multi-resolution analysis using wavelet transforms</li>
        </ul>
        
        <p>
            The extracted features are then used for classification through neural networks and other machine learning models, 
            enabling the identification of different tissue types based on their characteristic wave patterns.
        </p>
        
        <p>
            This application serves as an educational tool for understanding medical signal processing techniques and 
            as a demonstration of how these methods can be applied in practical scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown('<h3 class="glow-text">Getting Started</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>To begin using this application, follow these steps:</p>
        
        <ol>
            <li>Navigate to <strong>Wave Simulator</strong> to generate signal data for different tissue types</li>
            <li>Visit <strong>Signal Analyzer</strong> to perform detailed analysis in multiple domains</li>
            <li>Use <strong>Signal Classifier</strong> to identify tissue types using AI models</li>
        </ol>
        
        <p>
            Each page provides interactive controls for configuring parameters and exploring different aspects of 
            wave signal analysis. Experiment with different settings to see how they affect the analysis results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; color: #aaa; font-size: 0.8em;">
        Wave Feature Extractor for Medical Imaging Applications | Created for Signal Processing Mini-Project
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()