import unittest
import numpy as np
from scipy import signal
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wave_extractor import WaveFeatureExtractor

class TestWaveFeatureExtractor(unittest.TestCase):
    
    def setUp(self):
        # Create an extractor with a fixed sampling rate for testing
        self.sampling_rate = 100
        self.extractor = WaveFeatureExtractor(sampling_rate=self.sampling_rate)
        
        # Generate a sample signal for testing
        self.duration = 1.0
        self.test_frequencies = [5, 10, 20]
        self.test_noise_level = 0.1
        self.t, self.x = self.extractor.generate_sample_data(
            duration=self.duration,
            frequencies=self.test_frequencies,
            noise_level=self.test_noise_level
        )
    
    def test_initialization(self):
        # Test that the extractor initializes correctly
        self.assertEqual(self.extractor.sampling_rate, self.sampling_rate)
        self.assertIsInstance(self.extractor.results, dict)
    
    def test_generate_sample_data(self):
        # Test signal generation
        t, x = self.extractor.generate_sample_data(
            duration=self.duration,
            frequencies=self.test_frequencies,
            noise_level=self.test_noise_level
        )
        
        # Check that t and x have the expected lengths
        expected_length = int(self.sampling_rate * self.duration)
        self.assertEqual(len(t), expected_length)
        self.assertEqual(len(x), expected_length)
        
        # Check that the signal has the expected mean and standard deviation
        self.assertAlmostEqual(np.mean(x), 0.0, delta=0.5)  # Allow for noise
        
        # Check that parameters were stored correctly
        self.assertEqual(self.extractor.results['parameters']['duration'], self.duration)
        self.assertEqual(self.extractor.results['parameters']['frequencies'], self.test_frequencies)
        self.assertEqual(self.extractor.results['parameters']['noise_level'], self.test_noise_level)
    
    def test_generate_simulated_medical_wave(self):
        # Test medical wave simulation for different tissue types
        tissue_types = ["normal", "abnormal", "tumor", "cyst"]
        
        for tissue_type in tissue_types:
            t, x = self.extractor.generate_simulated_medical_wave(
                duration=self.duration,
                tissue_type=tissue_type,
                abnormality_level=0.5
            )
            
            # Check that t and x have the expected lengths
            expected_length = int(self.sampling_rate * self.duration)
            self.assertEqual(len(t), expected_length)
            self.assertEqual(len(x), expected_length)
            
            # Check that parameters were stored correctly
            self.assertEqual(self.extractor.results['parameters']['tissue_type'], tissue_type)
            self.assertEqual(self.extractor.results['parameters']['abnormality_level'], 0.5)
    
    def test_apply_fourier_transform(self):
        # Test FFT functionality
        freqs, fft_mag = self.extractor.apply_fourier_transform(self.x)
        
        # Check that frequencies and magnitude have expected lengths
        expected_freq_length = len(self.x) // 2 + 1  # For real FFT
        self.assertEqual(len(freqs), expected_freq_length)
        self.assertEqual(len(fft_mag), expected_freq_length)
        
        # Check that frequencies are in the expected range
        self.assertGreaterEqual(freqs[0], 0.0)
        self.assertLessEqual(freqs[-1], self.sampling_rate / 2)
        
        # Check that the FFT results were stored
        self.assertIn('fft', self.extractor.results)
        self.assertIn('frequencies', self.extractor.results['fft'])
        self.assertIn('magnitude', self.extractor.results['fft'])
    
    def test_apply_stft(self):
        # Test Short-Time Fourier Transform
        f, t_bins, Zxx = self.extractor.apply_stft(self.x, window_size=25, overlap=12)
        
        # Check that the dimensions are correct
        self.assertEqual(len(f), 13)  # Depends on the window size (25//2 + 1)
        
        # Check that the STFT results were stored
        self.assertIn('stft', self.extractor.results)
        self.assertIn('frequencies', self.extractor.results['stft'])
        self.assertIn('time_bins', self.extractor.results['stft'])
        self.assertIn('spectrogram', self.extractor.results['stft'])
    
    def test_apply_wavelet_transform(self):
        # Test wavelet transform
        coeffs = self.extractor.apply_wavelet_transform(self.x, wavelet='db4', level=3)
        
        # Check that we got the expected number of coefficient arrays
        self.assertEqual(len(coeffs), 4)  # 1 approximation + 3 detail coefficients
        
        # Check that the results were stored
        self.assertIn('wavelet', self.extractor.results)
        self.assertIn('coefficients', self.extractor.results['wavelet'])
        self.assertEqual(self.extractor.results['wavelet']['wavelet_type'], 'db4')
        self.assertEqual(self.extractor.results['wavelet']['level'], 3)
    
    def test_extract_wavelet_features(self):
        # Test wavelet feature extraction
        features = self.extractor.extract_wavelet_features(self.x, wavelet='db4', level=3)
        
        # Check that we have features for each coefficient level
        expected_features = ['cA_mean', 'cA_std', 'cA_energy', 'cA_entropy']
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Check detail coefficient features for each level
        for i in range(3):
            level_name = f'cD{3 - i}'
            expected_level_features = [
                f'{level_name}_mean', 
                f'{level_name}_std', 
                f'{level_name}_energy', 
                f'{level_name}_entropy'
            ]
            for feature in expected_level_features:
                self.assertIn(feature, features)
    
    def test_extract_frequency_features(self):
        # Apply FFT first
        self.extractor.apply_fourier_transform(self.x)
        
        # Test frequency feature extraction
        features = self.extractor.extract_frequency_features(self.x, n_peaks=3)
        
        # Check that we have the expected features
        expected_features = [
            'spectral_mean', 'spectral_std', 'spectral_skewness', 'spectral_kurtosis',
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Check peak features
        for i in range(3):
            self.assertIn(f'peak_{i+1}_freq', features)
            self.assertIn(f'peak_{i+1}_magnitude', features)
    
    def test_extract_time_domain_features(self):
        # Test time domain feature extraction
        features = self.extractor.extract_time_domain_features(self.x)
        
        # Check that we have the expected features
        expected_features = [
            'mean', 'std', 'min', 'max', 'range', 'rms',
            'skewness', 'kurtosis', 'num_peaks', 'mean_peak_height',
            'peak_interval', 'zero_crossing_rate'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
    
    def test_extract_all_features(self):
        # Test extraction of all features
        all_features = self.extractor.extract_all_features(
            self.x, 
            wavelet='db4', 
            wavelet_level=3, 
            n_peaks=3
        )
        
        # Check that features from all domains are present
        # (We'll just check a few examples from each domain)
        self.assertIn('time_mean', all_features)
        self.assertIn('freq_spectral_centroid', all_features)
        self.assertIn('wavelet_cA_energy', all_features)
        
        # Check that the combined features were stored
        self.assertIn('all_features', self.extractor.results)
    
    def test_feature_consistency(self):
        # Generate two identical signals and verify features are the same
        t1, x1 = self.extractor.generate_sample_data(
            duration=self.duration,
            frequencies=self.test_frequencies,
            noise_level=0.0  # No noise for deterministic output
        )
        
        # Reset the extractor to clear results
        self.extractor = WaveFeatureExtractor(sampling_rate=self.sampling_rate)
        
        t2, x2 = self.extractor.generate_sample_data(
            duration=self.duration,
            frequencies=self.test_frequencies,
            noise_level=0.0  # No noise for deterministic output
        )
        
        # Extract features from both signals
        features1 = self.extractor.extract_all_features(x1)
        
        # Reset the extractor again
        self.extractor = WaveFeatureExtractor(sampling_rate=self.sampling_rate)
        
        features2 = self.extractor.extract_all_features(x2)
        
        # Check that features are identical
        for key in features1:
            self.assertAlmostEqual(features1[key], features2[key], places=5)

if __name__ == '__main__':
    unittest.main()