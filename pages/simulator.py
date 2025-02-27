import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import uuid

from modules.wave_extractor import WaveFeatureExtractor
from modules.utils import (
    apply_custom_css,
    card,
    st_matplotlib_figure,
    loading_animation,
    visualize_signal_comparison,
    generate_download_link
)

# Set page configuration
st.set_page_config(
    page_title="Wave Feature Extractor - Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state variables if they don't exist
if 'extractor' not in st.session_state:
    st.session_state.extractor = WaveFeatureExtractor(sampling_rate=1000)
if 'simulator_signals' not in st.session_state:
    st.session_state.simulator_signals = {}

def main():
    # Page title
    st.markdown('<h1 class="glow-text">Wave Simulator</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="card">
        <p>
            The Wave Simulator allows you to generate and save simulated wave signals that mimic 
            different tissue types in medical imaging. You can adjust various parameters to create 
            signals with specific characteristics for analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## Signal Parameters")
        
        # Basic parameters
        st.markdown("### Basic Parameters")
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
        
        duration = st.slider(
            "Duration (seconds)", 
            min_value=0.5, 
            max_value=5.0, 
            value=1.0,
            step=0.5
        )
        
        sampling_rate = st.slider(
            "Sampling Rate (Hz)", 
            min_value=500, 
            max_value=2000, 
            value=1000,
            step=100
        )
        
        # Advanced parameters
        st.markdown("### Advanced Parameters")
        show_advanced = st.checkbox("Show Advanced Parameters", value=False)
        
        if show_advanced:
            noise_level_multiplier = st.slider(
                "Noise Level Multiplier", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                step=0.1
            )
            
            frequency_multiplier = st.slider(
                "Frequency Multiplier", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                step=0.1
            )
            
            amplitude_multiplier = st.slider(
                "Amplitude Multiplier", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                step=0.1
            )
        else:
            noise_level_multiplier = 1.0
            frequency_multiplier = 1.0
            amplitude_multiplier = 1.0
        
        # Generate button
        if st.button("Generate Signal", key="generate_signal_button"):
            with st.spinner("Generating signal..."):
                # Create extractor with specified sampling rate
                extractor = WaveFeatureExtractor(sampling_rate=sampling_rate)
                
                # Store advanced parameters
                extractor.advanced_params = {
                    'noise_level_multiplier': noise_level_multiplier,
                    'frequency_multiplier': frequency_multiplier,
                    'amplitude_multiplier': amplitude_multiplier
                }
                
                # Generate the wave
                t, x = extractor.generate_simulated_medical_wave(
                    duration=duration,
                    tissue_type=tissue_type,
                    abnormality_level=abnormality_level
                )
                
                # Apply additional advanced parameters
                if noise_level_multiplier != 1.0:
                    # Add extra noise
                    x += np.random.normal(0, abnormality_level * 0.1 * noise_level_multiplier, len(x))
                
                if frequency_multiplier != 1.0:
                    # This is a simplified way to "stretch" the signal in frequency domain
                    # For complex signals, would need to use FFT and manipulate frequencies
                    t_mod = t * frequency_multiplier
                    x_mod = np.interp(t, t_mod, x)
                    x = x_mod
                
                if amplitude_multiplier != 1.0:
                    # Scale amplitude
                    x = x * amplitude_multiplier
                
                # Store in session state
                st.session_state.extractor = extractor
                st.session_state.current_signal = x
                st.session_state.current_time = t
                st.session_state.current_tissue_type = tissue_type
                st.session_state.current_abnormality = abnormality_level
                st.session_state.simulator_signal_generated = True
        
        # Save signal section
        if 'simulator_signal_generated' in st.session_state and st.session_state.simulator_signal_generated:
            st.markdown("### Save Signal")
            
            signal_name = st.text_input(
                "Signal Name",
                value=f"{tissue_type.capitalize()}_Signal_{int(time.time())}"
            )
            
            if st.button("Save Signal", key="save_signal_button"):
                if signal_name in st.session_state.simulator_signals:
                    st.warning(f"Signal '{signal_name}' already exists. Please choose a different name.")
                else:
                    # Save the signal to session state
                    st.session_state.simulator_signals[signal_name] = {
                        'time': st.session_state.current_time,
                        'signal': st.session_state.current_signal,
                        'tissue_type': st.session_state.current_tissue_type,
                        'abnormality_level': st.session_state.current_abnormality,
                        'sampling_rate': st.session_state.extractor.sampling_rate,
                        'advanced_params': getattr(st.session_state.extractor, 'advanced_params', {})
                    }
                    
                    st.success(f"Signal '{signal_name}' saved successfully!")
    
    # Main content area
    if 'simulator_signal_generated' in st.session_state and st.session_state.simulator_signal_generated:
        # Signal visualization
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create visualization of the current signal
            fig = plt.figure(figsize=(10, 5), facecolor='#0e1117')
            ax = fig.add_subplot(111)
            
            t = st.session_state.current_time
            x = st.session_state.current_signal
            
            ax.plot(t, x, color='#4c83ff', linewidth=1.5)
            ax.set_title('Generated Signal', color='white', fontsize=14)
            ax.set_xlabel('Time (s)', color='white')
            ax.set_ylabel('Amplitude', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#0e1117')
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Signal download options
            st.markdown('<div class="card"><h3 class="glow-text">Download Options</h3>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # CSV download
                csv_data = pd.DataFrame({
                    'time': t,
                    'amplitude': x
                }).to_csv(index=False)
                
                generate_download_link(
                    csv_data, 
                    f"{st.session_state.current_tissue_type}_signal.csv", 
                    "Download CSV"
                )
            
            with col_b:
                # NumPy array download
                np_data = np.column_stack((t, x))
                np_bytes = io.BytesIO()
                np.save(np_bytes, np_data)
                np_bytes.seek(0)
                
                generate_download_link(
                    np_bytes.getvalue(), 
                    f"{st.session_state.current_tissue_type}_signal.npy", 
                    "Download NumPy Array",
                    mime="application/octet-stream"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Signal information
            st.markdown(f"""
            <div class="card">
                <h3 class="glow-text">Signal Properties</h3>
                <p><strong>Tissue Type:</strong> {st.session_state.current_tissue_type.capitalize()}</p>
                <p><strong>Abnormality Level:</strong> {st.session_state.current_abnormality:.2f}</p>
                <p><strong>Duration:</strong> {duration} seconds</p>
                <p><strong>Sampling Rate:</strong> {sampling_rate} Hz</p>
                <p><strong>Samples:</strong> {len(x)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advanced parameters if used
            if show_advanced:
                st.markdown(f"""
                <div class="card">
                    <h3 class="glow-text">Advanced Properties</h3>
                    <p><strong>Noise Level Multiplier:</strong> {noise_level_multiplier:.1f}</p>
                    <p><strong>Frequency Multiplier:</strong> {frequency_multiplier:.1f}</p>
                    <p><strong>Amplitude Multiplier:</strong> {amplitude_multiplier:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Signal catalog section
        st.markdown('<h3 class="glow-text">Signal Catalog</h3>', unsafe_allow_html=True)
        
        if st.session_state.simulator_signals:
            # Display saved signals in a grid
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Create a grid of signals
            cols = st.columns(3)
            
            for i, (name, signal_data) in enumerate(st.session_state.simulator_signals.items()):
                col_idx = i % 3
                
                with cols[col_idx]:
                    st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
                    
                    # Create a small preview plot
                    fig = plt.figure(figsize=(5, 3), facecolor='#0e1117')
                    ax = fig.add_subplot(111)
                    
                    t = signal_data['time']
                    x = signal_data['signal']
                    
                    ax.plot(t, x, color='#4c83ff', linewidth=1)
                    ax.set_title(
                        f"{signal_data['tissue_type'].capitalize()}, Abn: {signal_data['abnormality_level']:.1f}",
                        color='white',
                        fontsize=10
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor('#0e1117')
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Load button - applies the signal to the analyzer or classifier
                        if st.button("Load", key=f"load_{i}"):
                            st.session_state.current_signal = signal_data['signal']
                            st.session_state.current_time = signal_data['time']
                            st.session_state.current_tissue_type = signal_data['tissue_type']
                            st.session_state.current_abnormality = signal_data['abnormality_level']
                            st.session_state.analyzer_signal_generated = True
                            
                            # Show success message
                            st.success(f"Signal '{name}' loaded!")
                    
                    with col_b:
                        # Delete button
                        if st.button("Delete", key=f"delete_{i}"):
                            del st.session_state.simulator_signals[name]
                            st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No signals saved yet. Generate and save signals to add them to your catalog.")
        
        # Comparison section
        if len(st.session_state.simulator_signals) >= 2:
            st.markdown('<h3 class="glow-text">Signal Comparison</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Signal selection for comparison
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                signal_options = list(st.session_state.simulator_signals.keys())
                
                signal1 = st.selectbox(
                    "Signal 1",
                    signal_options,
                    index=0,
                    key="compare_signal1"
                )
                
                signal2 = st.selectbox(
                    "Signal 2",
                    signal_options,
                    index=min(1, len(signal_options)-1),
                    key="compare_signal2"
                )
                
                if st.button("Compare Signals"):
                    if signal1 == signal2:
                        st.warning("Please select two different signals to compare.")
                    else:
                        st.session_state.comparing_signals = True
                        st.session_state.signal1_name = signal1
                        st.session_state.signal2_name = signal2
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col1:
                # Display comparison if signals are selected
                if 'comparing_signals' in st.session_state and st.session_state.comparing_signals:
                    signal1_data = st.session_state.simulator_signals[st.session_state.signal1_name]
                    signal2_data = st.session_state.simulator_signals[st.session_state.signal2_name]
                    
                    # Create comparison visualization
                    fig = visualize_signal_comparison(
                        signal1_data['signal'],
                        signal2_data['signal'],
                        labels=[st.session_state.signal1_name, st.session_state.signal2_name],
                        sampling_rate=max(signal1_data['sampling_rate'], signal2_data['sampling_rate'])
                    )
                    
                    st.pyplot(fig)
                    
                    # Show comparison info
                    st.markdown('<div class="card"><h4 class="glow-text">Comparison Details</h4>', unsafe_allow_html=True)
                    
                    # Create a comparison table
                    comparison_data = {
                        'Property': ['Tissue Type', 'Abnormality Level', 'Sampling Rate', 'Duration', 'Samples'],
                        st.session_state.signal1_name: [
                            signal1_data['tissue_type'].capitalize(),
                            f"{signal1_data['abnormality_level']:.2f}",
                            f"{signal1_data['sampling_rate']} Hz",
                            f"{len(signal1_data['signal'])/signal1_data['sampling_rate']:.2f} s",
                            len(signal1_data['signal'])
                        ],
                        st.session_state.signal2_name: [
                            signal2_data['tissue_type'].capitalize(),
                            f"{signal2_data['abnormality_level']:.2f}",
                            f"{signal2_data['sampling_rate']} Hz",
                            f"{len(signal2_data['signal'])/signal2_data['sampling_rate']:.2f} s",
                            len(signal2_data['signal'])
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.table(comparison_df)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Select two signals and click 'Compare Signals' to see a comparison.")
    else:
        # No signal generated yet
        st.info("Adjust the parameters in the sidebar and click 'Generate Signal' to create a signal.")
        
        # Display explanation about different tissue types
        st.markdown('<h3 class="glow-text">About Tissue Types</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>
                This simulator generates synthetic wave signals that mimic how different tissue types might 
                respond in medical imaging applications. Here's what each tissue type represents:
            </p>
            
            <h4>Normal Tissue</h4>
            <p>
                Normal tissue shows regular, predictable patterns with low noise and consistent frequency 
                responses. The wave pattern is characterized by smooth transitions and low-amplitude reflections.
            </p>
            
            <h4>Abnormal Tissue</h4>
            <p>
                Abnormal tissue exhibits slight deviations from normal patterns, with increased noise and 
                more pronounced reflections. The abnormality level determines how much the signal deviates 
                from the normal pattern.
            </p>
            
            <h4>Tumor</h4>
            <p>
                Tumor signals display distinctive high-amplitude reflections at specific frequency ranges. 
                As the abnormality level increases, these reflections become more pronounced, mimicking 
                the increased density often found in tumorous tissue.
            </p>
            
            <h4>Cyst</h4>
            <p>
                Cyst signals feature characteristic periodic patterns with specific frequency signatures. 
                They typically show smoother transitions than tumors but with distinctive resonance patterns 
                that differentiate them from normal tissue.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display usage instructions
        st.markdown('<h3 class="glow-text">How to Use</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <ol>
                <li>Select a tissue type from the dropdown in the sidebar</li>
                <li>Adjust the abnormality level to control how pronounced the characteristics are</li>
                <li>Set the duration and sampling rate as needed</li>
                <li>For more control, enable "Show Advanced Parameters" to adjust noise, frequency, and amplitude</li>
                <li>Click "Generate Signal" to create the wave</li>
                <li>Save generated signals to your catalog for later use in analysis or classification</li>
                <li>Compare different signals to visualize their differences</li>
            </ol>
            
            <p>
                Generated signals can be loaded into the Analyzer and Classifier pages for further analysis 
                and tissue type identification.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()