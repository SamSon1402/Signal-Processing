import numpy as np
from scipy import signal
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class WaveFeatureExtractor:
    """
    A class for extracting and visualizing features from wave-based data,
    with applications in medical imaging and signal processing.
    """
    
    def __init__(self, sampling_rate=1000):
        """
        Initialize the feature extractor with a sampling rate.
        
        Parameters:
        -----------
        sampling_rate : int
            The sampling rate of the signal in Hz
        """
        self.sampling_rate = sampling_rate
        self.results = {}
        
    def generate_sample_data(self, duration=1.0, frequencies=[5, 10, 50], noise_level=0.1):
        """
        Generate sample wave data containing multiple frequencies and noise.
        
        Parameters:
        -----------
        duration : float
            Duration of the signal in seconds
        frequencies : list
            List of frequencies to include in the signal in Hz
        noise_level : float
            Standard deviation of Gaussian noise to add
            
        Returns:
        --------
        t : numpy.ndarray
            Time points
        x : numpy.ndarray
            Generated signal
        """
        # Create time points
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Initialize signal
        x = np.zeros_like(t)
        
        # Add component frequencies
        for freq in frequencies:
            x += np.sin(2 * np.pi * freq * t)
            
        # Add Gaussian noise
        x += np.random.normal(0, noise_level, len(t))
        
        # Store results
        self.results['time'] = t
        self.results['signal'] = x
        self.results['parameters'] = {
            'duration': duration,
            'frequencies': frequencies,
            'noise_level': noise_level
        }
        
        return t, x
    
    def generate_simulated_medical_wave(self, duration=1.0, tissue_type="normal", abnormality_level=0):
        """
        Generate simulated waves that mimic medical tissue responses
        
        Parameters:
        -----------
        duration : float
            Duration of the signal in seconds
        tissue_type : str
            Type of tissue ("normal", "abnormal", "tumor", "cyst")
        abnormality_level : float
            Level of abnormality (0-1, where 0 is normal, 1 is severe)
        """
        # Create time points
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Initialize signal
        x = np.zeros_like(t)
        
        # Base frequencies for all tissue types
        base_freqs = [5, 15, 30]
        base_amps = [1.0, 0.7, 0.4]
        base_phases = [0, np.pi/6, np.pi/3]
        
        # Add base components to all signals
        for freq, amp, phase in zip(base_freqs, base_amps, base_phases):
            x += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Tissue-specific characteristics
        if tissue_type == "normal":
            reflection_freqs = [60, 80, 100]
            reflection_amps = [0.05, 0.04, 0.03]
            reflection_phases = [np.pi/4, np.pi/2, 3*np.pi/4]
            noise_level = 0.1 + 0.05 * abnormality_level
        
        elif tissue_type == "abnormal":
            reflection_freqs = [55, 75, 95]
            reflection_amps = [0.1 + 0.1 * abnormality_level, 
                               0.08 + 0.07 * abnormality_level, 
                               0.06 + 0.04 * abnormality_level]
            reflection_phases = [np.pi/6, np.pi/3, 2*np.pi/3]
            noise_level = 0.15 + 0.1 * abnormality_level
            
            # Add characteristic reflection pattern
            for i in range(5):
                pos = int(self.sampling_rate * duration * (0.2 + 0.1 * i))
                if pos < len(x):
                    x[pos:pos+50] += 0.2 * abnormality_level * np.exp(-np.arange(50)/10)
        
        elif tissue_type == "tumor":
            reflection_freqs = [50, 70, 90]
            reflection_amps = [0.2 + 0.2 * abnormality_level, 
                               0.15 + 0.15 * abnormality_level, 
                               0.1 + 0.1 * abnormality_level]
            reflection_phases = [np.pi/5, np.pi/4, 3*np.pi/4]
            noise_level = 0.2 + 0.15 * abnormality_level
            
            # Add strong reflection pattern for tumors
            pos = int(self.sampling_rate * duration * 0.4)
            if pos < len(x):
                reflection_width = int(self.sampling_rate * 0.1)
                reflection_pattern = 0.3 * abnormality_level * np.exp(-np.arange(reflection_width)/20)
                if pos + reflection_width <= len(x):
                    x[pos:pos+reflection_width] += reflection_pattern
                else:
                    x[pos:] += reflection_pattern[:len(x)-pos]
        
        elif tissue_type == "cyst":
            reflection_freqs = [45, 65, 85]
            reflection_amps = [0.15 + 0.15 * abnormality_level, 
                               0.12 + 0.1 * abnormality_level, 
                               0.08 + 0.07 * abnormality_level]
            reflection_phases = [np.pi/7, np.pi/5, 2*np.pi/5]
            noise_level = 0.1 + 0.1 * abnormality_level
            
            # Add periodic pattern for cysts
            cyst_freq = 40 + 10 * abnormality_level
            x += 0.2 * abnormality_level * np.sin(2 * np.pi * cyst_freq * t + np.pi/3)
        
        # Add tissue-specific reflection patterns
        for freq, amp, phase in zip(reflection_freqs, reflection_amps, reflection_phases):
            x += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add Gaussian noise
        x += np.random.normal(0, noise_level, len(t))
        
        # Store results
        self.results['time'] = t
        self.results['signal'] = x
        self.results['parameters'] = {
            'duration': duration,
            'tissue_type': tissue_type,
            'abnormality_level': abnormality_level
        }
        
        return t, x
    
    def apply_fourier_transform(self, signal=None):
        """
        Apply Fast Fourier Transform to extract frequency components.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
            
        Returns:
        --------
        freqs : numpy.ndarray
            Frequency bins
        fft_mag : numpy.ndarray
            Magnitude of the FFT
        """
        if signal is None and 'signal' in self.results:
            signal = self.results['signal']
        elif signal is None:
            raise ValueError("No signal provided or stored")
            
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        fft_mag = np.abs(fft_result)
        
        # Compute frequency bins
        n = len(signal)
        freqs = np.fft.rfftfreq(n, 1/self.sampling_rate)
        
        # Store results
        self.results['fft'] = {
            'frequencies': freqs,
            'magnitude': fft_mag,
            'complex': fft_result
        }
        
        return freqs, fft_mag
    
    def apply_stft(self, signal=None, window_size=256, overlap=128):
        """
        Apply Short-Time Fourier Transform for time-frequency analysis.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
        window_size : int
            Size of the window for STFT
        overlap : int
            Overlap between consecutive windows
            
        Returns:
        --------
        f : numpy.ndarray
            Frequency bins
        t : numpy.ndarray
            Time bins
        Zxx : numpy.ndarray
            STFT result (complex spectrogram)
        """
        if signal is None and 'signal' in self.results:
            signal = self.results['signal']
            t = self.results['time']
        elif signal is None:
            raise ValueError("No signal provided or stored")
        else:
            # Assuming signal is provided without time points
            t = np.arange(len(signal)) / self.sampling_rate
            
        # Compute STFT
        f, t_bins, Zxx = signal.stft(signal, self.sampling_rate, 
                                    nperseg=window_size, 
                                    noverlap=overlap)
        
        # Store results
        self.results['stft'] = {
            'frequencies': f,
            'time_bins': t_bins,
            'spectrogram': Zxx
        }
        
        return f, t_bins, Zxx
    
    def apply_wavelet_transform(self, signal=None, wavelet='db4', level=5):
        """
        Apply Discrete Wavelet Transform for multi-resolution analysis.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
        wavelet : str
            Wavelet type to use (e.g., 'db4', 'sym5')
        level : int
            Decomposition level
            
        Returns:
        --------
        coeffs : list
            Wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD1]
            where cA_n is approximation coefficient and cD_n is detail coefficient
        """
        if signal is None and 'signal' in self.results:
            signal = self.results['signal']
        elif signal is None:
            raise ValueError("No signal provided or stored")
            
        # Apply wavelet transform
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Store results
        self.results['wavelet'] = {
            'coefficients': coeffs,
            'wavelet_type': wavelet,
            'level': level
        }
        
        return coeffs
    
    def extract_wavelet_features(self, signal=None, wavelet='db4', level=5):
        """
        Extract statistical features from wavelet coefficients.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
        wavelet : str
            Wavelet type to use
        level : int
            Decomposition level
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Apply wavelet transform if not already done
        if 'wavelet' not in self.results or self.results['wavelet']['wavelet_type'] != wavelet:
            coeffs = self.apply_wavelet_transform(signal, wavelet, level)
        else:
            coeffs = self.results['wavelet']['coefficients']
            
        # Extract features from each coefficient level
        features = {}
        
        # Approximation coefficients
        cA = coeffs[0]
        features['cA_mean'] = np.mean(cA)
        features['cA_std'] = np.std(cA)
        features['cA_energy'] = np.sum(cA**2)
        features['cA_entropy'] = -np.sum(cA**2 * np.log(cA**2 + 1e-10))
        
        # Detail coefficients
        for i in range(level):
            cD = coeffs[i + 1]
            level_name = f'cD{level - i}'
            
            features[f'{level_name}_mean'] = np.mean(cD)
            features[f'{level_name}_std'] = np.std(cD)
            features[f'{level_name}_energy'] = np.sum(cD**2)
            features[f'{level_name}_entropy'] = -np.sum(cD**2 * np.log(cD**2 + 1e-10))
            
        # Store features
        self.results['wavelet_features'] = features
        
        return features
    
    def extract_frequency_features(self, signal=None, n_peaks=5):
        """
        Extract features from the frequency domain.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
        n_peaks : int
            Number of dominant frequency peaks to extract
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        # Apply FFT if not already done
        if 'fft' not in self.results:
            freqs, fft_mag = self.apply_fourier_transform(signal)
        else:
            freqs = self.results['fft']['frequencies']
            fft_mag = self.results['fft']['magnitude']
            
        # Extract features
        features = {}
        
        # Basic statistical features
        features['spectral_mean'] = np.mean(fft_mag)
        features['spectral_std'] = np.std(fft_mag)
        features['spectral_skewness'] = np.mean(((fft_mag - np.mean(fft_mag)) / (np.std(fft_mag) + 1e-10))**3)
        features['spectral_kurtosis'] = np.mean(((fft_mag - np.mean(fft_mag)) / (np.std(fft_mag) + 1e-10))**4)
        
        # Spectral centroid (weighted average of frequencies)
        features['spectral_centroid'] = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_mag) / (np.sum(fft_mag) + 1e-10))
        
        # Spectral rolloff (frequency below which 85% of spectral energy lies)
        cumsum = np.cumsum(fft_mag)
        rolloff_point = 0.85 * cumsum[-1]
        features['spectral_rolloff'] = freqs[np.where(cumsum >= rolloff_point)[0][0]] if len(np.where(cumsum >= rolloff_point)[0]) > 0 else 0
        
        # Dominant frequencies
        peak_indices = np.argsort(fft_mag)[-n_peaks:]
        for i, idx in enumerate(peak_indices):
            features[f'peak_{i+1}_freq'] = freqs[idx]
            features[f'peak_{i+1}_magnitude'] = fft_mag[idx]
            
        # Store features
        self.results['frequency_features'] = features
        
        return features
    
    def extract_time_domain_features(self, signal=None):
        """
        Extract features from the time domain.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        if signal is None and 'signal' in self.results:
            signal = self.results['signal']
        elif signal is None:
            raise ValueError("No signal provided or stored")
            
        # Extract features
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # Shape features
        features['skewness'] = np.mean(((signal - features['mean']) / (features['std'] + 1e-10))**3) if features['std'] != 0 else 0
        features['kurtosis'] = np.mean(((signal - features['mean']) / (features['std'] + 1e-10))**4) if features['std'] != 0 else 0
        
        # Peak-related features
        peaks, _ = signal.find_peaks(signal, height=features['mean'])
        if len(peaks) > 0:
            features['num_peaks'] = len(peaks)
            features['mean_peak_height'] = np.mean(signal[peaks])
            features['peak_interval'] = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
        else:
            features['num_peaks'] = 0
            features['mean_peak_height'] = 0
            features['peak_interval'] = 0
            
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        # Store features
        self.results['time_features'] = features
        
        return features
    
    def extract_all_features(self, signal=None, wavelet='db4', wavelet_level=5, n_peaks=5):
        """
        Extract all available features from time, frequency, and wavelet domains.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Input signal (uses stored signal if None)
        wavelet : str
            Wavelet type to use
        wavelet_level : int
            Decomposition level for wavelet transform
        n_peaks : int
            Number of dominant frequency peaks to extract
            
        Returns:
        --------
        all_features : dict
            Dictionary of all extracted features
        """
        # Extract features from each domain
        time_features = self.extract_time_domain_features(signal)
        freq_features = self.extract_frequency_features(signal, n_peaks)
        wavelet_features = self.extract_wavelet_features(signal, wavelet, wavelet_level)
        
        # Combine all features
        all_features = {}
        all_features.update({f'time_{k}': v for k, v in time_features.items()})
        all_features.update({f'freq_{k}': v for k, v in freq_features.items()})
        all_features.update({f'wavelet_{k}': v for k, v in wavelet_features.items()})
        
        # Store combined features
        self.results['all_features'] = all_features
        
        return all_features
    
    def feature_importance_analysis(self, features_list):
        """
        Analyze the importance/variance of extracted features using PCA.
        
        Parameters:
        -----------
        features_list : list of dict
            List of feature dictionaries (one per signal)
            
        Returns:
        --------
        pca : sklearn.decomposition.PCA
            Fitted PCA model
        transformed_data : numpy.ndarray
            PCA-transformed data
        feature_importance : pandas.DataFrame
            DataFrame with feature importance scores
        """
        # Convert list of feature dictionaries to a DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle any missing values
        df.fillna(0, inplace=True)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Apply PCA
        pca = PCA()
        transformed_data = pca.fit_transform(scaled_data)
        
        # Create feature importance DataFrame
        components = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=df.columns
        )
        
        # Calculate feature importance
        loadings = components.copy()
        loadings['importance'] = np.abs(loadings['PC1']) 
        loadings = loadings.sort_values('importance', ascending=False)
        
        # Store results
        self.results['feature_analysis'] = {
            'pca': pca,
            'transformed_data': transformed_data,
            'feature_names': df.columns.tolist(),
            'feature_importance': loadings
        }
        
        return pca, transformed_data, loadings