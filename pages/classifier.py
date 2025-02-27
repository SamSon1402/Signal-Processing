import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import torch

from modules.wave_extractor import WaveFeatureExtractor
from modules.wave_classifier import WaveClassifier, load_pretrained_model
from modules.visualizations import (
    generate_basic_visualizations,
    generate_classification_result_visualization
)
from modules.utils import (
    apply_custom_css,
    card,
    st_matplotlib_figure,
    loading_animation,
    create_metric_card
)

# Set page configuration
st.set_page_config(
    page_title="Wave Feature Extractor - Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state variables if they don't exist
if 'extractor' not in st.session_state:
    st.session_state.extractor = WaveFeatureExtractor(sampling_rate=1000)
if 'classifier_signal_generated' not in st.session_state:
    st.session_state.classifier_signal_generated = False
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None

def main():
    # Page title
    st.markdown('<h1 class="glow-text">Signal Classifier</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="card">
        <p>
            The Signal Classifier uses machine learning models to identify tissue types based on their 
            wave signal patterns. You can generate new signals, use existing ones, or upload your own 
            data for classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## Classification Controls")
        
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
                    st.session_state.classifier_signal_generated = True
                    st.session_state.current_tissue_type = tissue_type
                    st.session_state.current_abnormality = abnormality_level
                    
                    # Reset classification results
                    if 'classification_results' in st.session_state:
                        del st.session_state.classification_results
        
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
                    st.session_state.classifier_signal_generated = True
                    st.session_state.current_tissue_type = signal_data.get('tissue_type', 'unknown')
                    st.session_state.current_abnormality = signal_data.get('abnormality_level', 0.0)
                    
                    # Reset classification results
                    if 'classification_results' in st.session_state:
                        del st.session_state.classification_results
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
                            st.session_state.classifier_signal_generated = True
                            st.session_state.current_tissue_type = "unknown"
                            st.session_state.current_abnormality = 0.0
                            
                            # Reset classification results
                            if 'classification_results' in st.session_state:
                                del st.session_state.classification_results
                    else:
                        st.error("CSV file must contain 'time' and 'amplitude' columns.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # Classification options
        if st.session_state.classifier_signal_generated:
            st.markdown("### Classification Options")
            
            model_type = st.selectbox(
                "Model Type",
                ["CNN", "Random Forest", "SVM"],
                index=0
            )
            
            use_feature_extraction = st.checkbox("Use Feature Extraction", value=False)
            
            if use_feature_extraction:
                # Feature extraction options
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
            
            # Classify button
            if st.button("Classify Signal", key="classify_button"):
                with st.spinner("Classifying signal..."):
                    loading_animation()
                    
                    # Extract features if needed
                    if use_feature_extraction:
                        extractor = st.session_state.extractor
                        features = extractor.extract_all_features(
                            signal=st.session_state.current_signal, 
                            wavelet=wavelet_type, 
                            wavelet_level=wavelet_level
                        )
                        st.session_state.extracted_features = features
                    
                    # Load and use the selected model
                    model = load_pretrained_model(model_type=model_type)
                    
                    # Prepare input based on model type
                    if model_type == "CNN":
                        # CNN takes raw signal
                        signal = st.session_state.current_signal
                        # Ensure length is appropriate
                        if len(signal) > 1000:
                            signal = signal[:1000]
                        elif len(signal) < 1000:
                            # Pad with zeros
                            signal = np.pad(signal, (0, 1000 - len(signal)), 'constant')
                        
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
                    else:
                        # RF and SVM use extracted features
                        if not use_feature_extraction:
                            # Extract features automatically
                            extractor = st.session_state.extractor
                            features = extractor.extract_all_features(signal=st.session_state.current_signal)
                            st.session_state.extracted_features = features
                        
                        # Convert features to numpy array
                        feature_values = np.array(list(st.session_state.extracted_features.values())).reshape(1, -1)
                        
                        # Make prediction
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(feature_values)[0]
                        else:
                            # If the model doesn't support predict_proba, use decision function
                            decision_values = model.decision_function(feature_values)
                            # Convert to probabilities using softmax
                            probs = np.exp(decision_values) / np.sum(np.exp(decision_values))
                        
                        st.session_state.classification_results = {
                            "normal": float(probs[0]),
                            "abnormal": float(probs[1]),
                            "tumor": float(probs[2]),
                            "cyst": float(probs[3])
                        }
                    
                    # Store model type for display
                    st.session_state.model_used = model_type
    
    # Main content area
    if st.session_state.classifier_signal_generated:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Signal visualization
            # Generate basic visualizations
            visualizations = generate_basic_visualizations(
                st.session_state.extractor, 
                st.session_state.current_time, 
                st.session_state.current_signal
            )
            
            # Show time domain signal
            st.image(visualizations['time'], use_column_width=True)
            
            # Advanced visualizations in tabs
            if 'classification_results' in st.session_state:
                tabs = st.tabs(["Frequency Domain", "Wavelet", "Features"])
                
                with tabs[0]:
                    st.image(visualizations['fft'], use_column_width=True)
                
                with tabs[1]:
                    st.image(visualizations['wavelet'], use_column_width=True)
                
                with tabs[2]:
                    if 'extracted_features' in st.session_state:
                        st.image(visualizations['features'], use_column_width=True)
                    else:
                        st.info("Extract features to view feature importance visualization.")
        
        with col2:
            # Signal information
            st.markdown(f"""
            <div class="card">
                <h3 class="glow-text">Signal Properties</h3>
                <p><strong>Source:</strong> {signal_source}</p>
                <p><strong>Signal Type:</strong> {st.session_state.current_tissue_type.capitalize()}</p>
                <p><strong>Abnormality Level:</strong> {st.session_state.current_abnormality:.2f}</p>
                <p><strong>Sampling Rate:</strong> {st.session_state.extractor.sampling_rate} Hz</p>
                <p><strong>Length:</strong> {len(st.session_state.current_signal)} samples</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification results
            if 'classification_results' in st.session_state:
                st.markdown('<div class="card"><h3 class="glow-text">Classification Results</h3>', unsafe_allow_html=True)
                
                # Create a bar chart for the classification probabilities
                results = st.session_state.classification_results
                fig = generate_classification_result_visualization(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the predicted class
                predicted_class = max(results, key=results.get)
                confidence = results[predicted_class] * 100
                
                # Accuracy indicator
                if st.session_state.current_tissue_type != "unknown":
                    is_correct = predicted_class == st.session_state.current_tissue_type
                    accuracy_color = "#4CAF50" if is_correct else "#F44336"
                    accuracy_icon = "✓" if is_correct else "✗"
                    
                    st.markdown(f"""
                    <p style="text-align:center; font-size:1.2em;">
                        Predicted class: <span style="color:#00f2ff; font-weight:bold;">{predicted_class.upper()}</span><br>
                        Confidence: <span style="color:#4c83ff; font-weight:bold;">{confidence:.1f}%</span><br>
                        Actual class: <span style="color:#ffffff; font-weight:bold;">{st.session_state.current_tissue_type.upper()}</span><br>
                        Accuracy: <span style="color:{accuracy_color}; font-weight:bold;">{accuracy_icon} {is_correct}</span>
                    </p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <p style="text-align:center; font-size:1.2em;">
                        Predicted class: <span style="color:#00f2ff; font-weight:bold;">{predicted_class.upper()}</span><br>
                        Confidence: <span style="color:#4c83ff; font-weight:bold;">{confidence:.1f}%</span>
                    </p>
                    """, unsafe_allow_html=True)
                
                # Model information
                st.markdown(f"""
                <div style="margin-top: 20px;">
                    <h4>Model Information</h4>
                    <p><strong>Model Type:</strong> {st.session_state.model_used}</p>
                    <p><strong>Feature Extraction:</strong> {'Yes' if 'extracted_features' in st.session_state else 'No'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Classification metrics
                st.markdown(f"""
                <div style="margin-top: 20px;">
                    <h4>Classification Metrics</h4>
                    <table style="width:100%; border-collapse: collapse;">
                        <tr>
                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #4c83ff;">Class</th>
                            <th style="text-align: right; padding: 8px; border-bottom: 1px solid #4c83ff;">Probability</th>
                        </tr>
                        <tr>
                            <td style="text-align: left; padding: 8px;">Normal</td>
                            <td style="text-align: right; padding: 8px;">{results['normal']*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="text-align: left; padding: 8px;">Abnormal</td>
                            <td style="text-align: right; padding: 8px;">{results['abnormal']*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="text-align: left; padding: 8px;">Tumor</td>
                            <td style="text-align: right; padding: 8px;">{results['tumor']*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="text-align: left; padding: 8px;">Cyst</td>
                            <td style="text-align: right; padding: 8px;">{results['cyst']*100:.2f}%</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Classification explanation
                st.markdown('<div class="card"><h3 class="glow-text">Interpretation</h3>', unsafe_allow_html=True)
                
                # Generate explanation based on predicted class and confidence
                if predicted_class == "normal":
                    explanation = """
                    <p>The signal has been classified as <strong>Normal Tissue</strong>. Normal tissue typically shows:</p>
                    <ul>
                        <li>Regular, predictable patterns</li>
                        <li>Low noise levels</li>
                        <li>Consistent frequency responses</li>
                        <li>Smooth transitions</li>
                        <li>Low-amplitude reflections</li>
                    </ul>
                    """
                elif predicted_class == "abnormal":
                    explanation = """
                    <p>The signal has been classified as <strong>Abnormal Tissue</strong>. Abnormal tissue typically shows:</p>
                    <ul>
                        <li>Slight deviations from normal patterns</li>
                        <li>Increased noise levels</li>
                        <li>More pronounced reflections</li>
                        <li>Variations in frequency response</li>
                        <li>Small irregularities in the waveform</li>
                    </ul>
                    """
                elif predicted_class == "tumor":
                    explanation = """
                    <p>The signal has been classified as <strong>Tumor</strong>. Tumor signals typically show:</p>
                    <ul>
                        <li>Distinctive high-amplitude reflections</li>
                        <li>Specific frequency signatures</li>
                        <li>Strong pattern distortions</li>
                        <li>Increased signal density</li>
                        <li>Sharp transitions in the waveform</li>
                    </ul>
                    """
                elif predicted_class == "cyst":
                    explanation = """
                    <p>The signal has been classified as <strong>Cyst</strong>. Cyst signals typically show:</p>
                    <ul>
                        <li>Characteristic periodic patterns</li>
                        <li>Specific frequency resonances</li>
                        <li>Smoother transitions than tumors</li>
                        <li>Distinctive echo patterns</li>
                        <li>Medium amplitude reflections</li>
                    </ul>
                    """
                
                # Add confidence-based commentary
                if confidence > 90:
                    confidence_note = "<p>The model has <strong>very high confidence</strong> in this classification.</p>"
                elif confidence > 75:
                    confidence_note = "<p>The model has <strong>high confidence</strong> in this classification.</p>"
                elif confidence > 50:
                    confidence_note = "<p>The model has <strong>moderate confidence</strong> in this classification.</p>"
                else:
                    confidence_note = "<p>The model has <strong>low confidence</strong> in this classification. The signal may have characteristics of multiple tissue types.</p>"
                
                st.markdown(explanation + confidence_note, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # No classification results yet
                st.info("Click 'Classify Signal' in the sidebar to analyze the signal and identify the tissue type.")
    
    else:
        # No signal generated yet
        st.info("Please generate or upload a signal using the controls in the sidebar.")
        
        # Display information about the classifier
        st.markdown('<h3 class="glow-text">About the Classifier</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>
                This classifier uses machine learning models to identify different tissue types based on their 
                characteristic wave signal patterns. The application includes three different model types:
            </p>
            
            <h4>Convolutional Neural Network (CNN)</h4>
            <p>
                The CNN model works directly with raw signal data, automatically learning relevant features 
                through its convolutional layers. It's particularly effective at identifying patterns across 
                different scales in the signal.
            </p>
            
            <h4>Random Forest</h4>
            <p>
                The Random Forest classifier uses extracted features from the time, frequency, and wavelet 
                domains. It excels at handling complex feature interactions and is less prone to overfitting.
            </p>
            
            <h4>Support Vector Machine (SVM)</h4>
            <p>
                The SVM classifier also works with extracted features and is effective at finding optimal 
                decision boundaries between different tissue types, especially with high-dimensional data.
            </p>
            
            <h4>Feature Extraction</h4>
            <p>
                When using Random Forest or SVM models, you can enable additional feature extraction to 
                customize the wavelet analysis parameters. This can improve classification accuracy for 
                certain signal types.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display usage instructions
        st.markdown('<h3 class="glow-text">How to Use</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <ol>
                <li>Select a signal source (generate new, use existing, or upload)</li>
                <li>Configure the signal parameters if generating a new signal</li>
                <li>Click "Generate Signal" or "Load Signal" to prepare the data</li>
                <li>Select a model type (CNN, Random Forest, or SVM)</li>
                <li>Optionally enable and configure feature extraction</li>
                <li>Click "Classify Signal" to run the classification</li>
                <li>View the results, including predicted class, confidence, and interpretation</li>
            </ol>
            
            <p>
                The classifier will analyze the signal and identify the most likely tissue type based on 
                the signal's characteristics. You can experiment with different models and feature extraction 
                settings to see how they affect the classification results.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()