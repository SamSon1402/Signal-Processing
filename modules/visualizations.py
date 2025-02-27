import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
import io

def st_matplotlib_figure(fig):
    """
    Convert a matplotlib figure to a PIL Image for display in Streamlit.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to convert
        
    Returns:
    --------
    image : PIL.Image
        The converted image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', 
                facecolor='#0e1117', edgecolor='none')
    buf.seek(0)
    return Image.open(buf)

def generate_basic_visualizations(extractor, t=None, x=None):
    """
    Generate basic visualizations for the signal.
    
    Parameters:
    -----------
    extractor : WaveFeatureExtractor
        The feature extractor
    t : numpy.ndarray, optional
        Time points 
    x : numpy.ndarray, optional
        Signal values
        
    Returns:
    --------
    visualizations : dict
        Dictionary containing visualization images
    """
    # Use provided signal or get from extractor
    if x is None:
        if 'signal' in extractor.results:
            x = extractor.results['signal']
            t = extractor.results['time']
        else:
            raise ValueError("No signal provided or stored in extractor")
    
    # Apply transforms if not already done
    if 'fft' not in extractor.results:
        freqs, fft_mag = extractor.apply_fourier_transform(x)
    else:
        freqs = extractor.results['fft']['frequencies']
        fft_mag = extractor.results['fft']['magnitude']
    
    if 'wavelet' not in extractor.results:
        coeffs = extractor.apply_wavelet_transform(x)
    else:
        coeffs = extractor.results['wavelet']['coefficients']
    
    # Extract features if requested
    if 'all_features' in extractor.results:
        features = extractor.results['all_features']
    else:
        features = None
    
    # Set figure style
    plt.style.use('dark_background')
    
    # 1. Time domain visualization
    fig_time = Figure(figsize=(10, 4))
    ax_time = fig_time.add_subplot(111)
    ax_time.plot(t, x, color='#4c83ff', linewidth=1.5)
    ax_time.set_title('Time Domain Signal', color='white', fontsize=14)
    ax_time.set_xlabel('Time (s)', color='white')
    ax_time.set_ylabel('Amplitude', color='white')
    ax_time.tick_params(colors='white')
    ax_time.grid(True, alpha=0.3)
    fig_time.tight_layout()
    
    # 2. Frequency domain visualization
    fig_fft = Figure(figsize=(10, 4))
    ax_fft = fig_fft.add_subplot(111)
    ax_fft.plot(freqs, fft_mag, color='#00f2ff', linewidth=1.5)
    ax_fft.set_title('Frequency Spectrum (FFT)', color='white', fontsize=14)
    ax_fft.set_xlabel('Frequency (Hz)', color='white')
    ax_fft.set_ylabel('Magnitude', color='white')
    ax_fft.tick_params(colors='white')
    ax_fft.grid(True, alpha=0.3)
    fig_fft.tight_layout()
    
    # 3. Wavelet visualization
    fig_wavelet = Figure(figsize=(10, 8))
    
    # Plot original signal
    ax_orig = fig_wavelet.add_subplot(len(coeffs) + 1, 1, 1)
    ax_orig.plot(x, color='#00c3ff', linewidth=1.5)
    ax_orig.set_title('Original Signal', color='white', fontsize=12)
    ax_orig.set_xticks([])
    ax_orig.tick_params(colors='white')
    
    # Plot approximation coefficients
    ax_approx = fig_wavelet.add_subplot(len(coeffs) + 1, 1, 2)
    ax_approx.plot(coeffs[0], color='#4c83ff', linewidth=1.5)
    ax_approx.set_title(f'Approximation Coefficients', color='white', fontsize=12)
    ax_approx.set_xticks([])
    ax_approx.tick_params(colors='white')
    
    # Plot detail coefficients
    for i in range(1, len(coeffs)):
        ax_detail = fig_wavelet.add_subplot(len(coeffs) + 1, 1, i + 2)
        ax_detail.plot(coeffs[i], color=f'#{int(255 - 40*i):02x}{int(100 + 25*i):02x}ff', linewidth=1.5)
        ax_detail.set_title(f'Detail Coefficients (Level {len(coeffs) - i})', color='white', fontsize=12)
        ax_detail.set_xticks([])
        ax_detail.tick_params(colors='white')
    
    fig_wavelet.tight_layout()
    
    # 4. Feature visualization (if features available)
    if features is not None:
        # Convert to a pandas Series for easier plotting
        features_series = pd.Series(features)
        
        # Sort by absolute value to see most prominent features
        sorted_features = features_series.abs().sort_values(ascending=False).index[:15]
        
        fig_features = Figure(figsize=(10, 6))
        ax_features = fig_features.add_subplot(111)
        
        # Get colormap
        values = features_series[sorted_features].values
        norm = plt.Normalize(values.min(), values.max())
        colors = plt.cm.viridis(norm(values))
        
        # Create horizontal bar chart
        bars = ax_features.barh(range(len(sorted_features)), 
                               features_series[sorted_features].abs(), 
                               color=colors)
        
        ax_features.set_yticks(range(len(sorted_features)))
        ax_features.set_yticklabels(sorted_features, fontsize=8)
        ax_features.set_title('Top 15 Feature Magnitudes', color='white', fontsize=14)
        ax_features.set_xlabel('Absolute Value', color='white')
        ax_features.tick_params(colors='white')
        fig_features.tight_layout()
    else:
        # Create an empty figure if no features
        fig_features = Figure(figsize=(10, 6))
        ax_features = fig_features.add_subplot(111)
        ax_features.text(0.5, 0.5, "No features extracted yet", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax_features.transAxes,
                       color='white',
                       fontsize=14)
        ax_features.set_xticks([])
        ax_features.set_yticks([])
        fig_features.tight_layout()
    
    # Return all visualizations as PIL images
    return {
        'time': st_matplotlib_figure(fig_time),
        'fft': st_matplotlib_figure(fig_fft),
        'wavelet': st_matplotlib_figure(fig_wavelet),
        'features': st_matplotlib_figure(fig_features)
    }

def generate_3d_visualization(extractor):
    """
    Generate 3D visualization of the time-frequency representation.
    
    Parameters:
    -----------
    extractor : WaveFeatureExtractor
        The feature extractor
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The 3D visualization figure
    """
    # Apply STFT if not already done
    if 'stft' not in extractor.results:
        f, t, Zxx = extractor.apply_stft()
    else:
        f = extractor.results['stft']['frequencies']
        t = extractor.results['stft']['time_bins']
        Zxx = extractor.results['stft']['spectrogram']
    
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
        title='3D Time-Frequency Representation',
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
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        scene_bgcolor='#111827',
        font=dict(color='white')
    )
    
    return fig

def generate_feature_comparison(extractor1, extractor2, labels=None):
    """
    Generate visualization comparing features between two signals.
    
    Parameters:
    -----------
    extractor1 : WaveFeatureExtractor
        First feature extractor
    extractor2 : WaveFeatureExtractor
        Second feature extractor
    labels : list, optional
        Labels for the two signals
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The comparison figure
    """
    # Extract features if not already done
    if 'all_features' not in extractor1.results:
        features1 = extractor1.extract_all_features()
    else:
        features1 = extractor1.results['all_features']
    
    if 'all_features' not in extractor2.results:
        features2 = extractor2.extract_all_features()
    else:
        features2 = extractor2.results['all_features']
    
    # Use default labels if none provided
    if labels is None:
        labels = ['Signal 1', 'Signal 2']
    
    # Find common features
    common_features = list(set(features1.keys()) & set(features2.keys()))
    
    # Sort by the absolute difference between the two signals
    feature_diff = {f: abs(features1[f] - features2[f]) for f in common_features}
    sorted_features = sorted(feature_diff.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 10 most different features
    top_features = [f for f, _ in sorted_features[:10]]
    
    # Create figure
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Bar positions
    bar_width = 0.35
    indices = np.arange(len(top_features))
    
    # Plot bars
    bars1 = ax.barh(indices - bar_width/2, [features1[f] for f in top_features], 
                   bar_width, label=labels[0], color='#4c83ff', alpha=0.8)
    bars2 = ax.barh(indices + bar_width/2, [features2[f] for f in top_features], 
                   bar_width, label=labels[1], color='#00f2ff', alpha=0.8)
    
    # Add labels and title
    ax.set_yticks(indices)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_title('Top Feature Differences', color='white', fontsize=14)
    ax.set_xlabel('Feature Value', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    
    # Add difference percentage
    for i, f in enumerate(top_features):
        if features1[f] != 0 and features2[f] != 0:
            diff_pct = abs(features1[f] - features2[f]) / max(abs(features1[f]), abs(features2[f])) * 100
            ax.text(max(features1[f], features2[f]) * 1.05, i, f"{diff_pct:.1f}%", 
                   verticalalignment='center', color='white')
    
    fig.tight_layout()
    
    return fig

def generate_classification_result_visualization(results):
    """
    Generate visualization for classification results.
    
    Parameters:
    -----------
    results : dict
        Classification probability results
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The visualization figure
    """
    # Create a bar chart for the classification probabilities
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
    
    return fig