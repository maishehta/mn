import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mne
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import os
import json
import yaml
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from functools import lru_cache
import hashlib
from pathlib import Path
import h5py
from tqdm import tqdm
import antropy as ant
from scipy.stats import entropy
from scipy.signal import hilbert
import pywt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class EEGConfig:
    """Configuration class for EEG processing parameters."""
    # Sampling parameters
    fs: int = 256
    expected_fs: int = 256
   
    # Filtering parameters
    lowcut: float = 0.5
    highcut: float = 50.0
    notch_freq: float = 60.0
    notch_quality: int = 30
    filter_order: int = 5
   
    # Segmentation parameters
    segment_length: float = 2.0
    overlap: float = 0.5
    min_segment_length: float = 1.0
   
    # Artifact removal parameters
    z_threshold: float = 4.0
    use_ica: bool = True
    ica_components: int = 6
   
    # Feature extraction parameters
    freq_bands: Dict[str, Tuple[float, float]] = None
   
    # Processing parameters
    use_car: bool = True  # Common Average Reference
    use_quality_check: bool = True
    chunk_size: int = 10000
    n_jobs: int = -1  # Use all available cores
   
    # Output parameters
    output_dir: str = "eeg_analysis"
    save_intermediate: bool = True
    plot_format: str = "png"
   
    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 50)
            }
   
    def save_config(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
   
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class EEGData:
    """Enhanced data structure for EEG data management."""
   
    def __init__(self, data: np.ndarray, timestamps: np.ndarray,
                 channels: List[str], fs: int, metadata: Dict = None):
        self.data = np.array(data)
        self.timestamps = np.array(timestamps)
        self.channels = channels
        self.fs = fs
        self.metadata = metadata or {}
        self.quality_metrics = {}
        self.artifacts = {}
        self._validate_data()
   
    def _validate_data(self):
        """Validate data integrity."""
        if self.data.shape[0] != len(self.timestamps):
            raise ValueError("Data and timestamps length mismatch")
        if self.data.shape[1] != len(self.channels):
            raise ValueError("Data channels and channel names mismatch")
        if np.any(np.isnan(self.data)):
            logging.warning("NaN values detected in data")
       
    def compute_quality_metrics(self):
        """Compute data quality metrics."""
        self.quality_metrics = {
            'snr': self._compute_snr(),
            'data_range': np.ptp(self.data, axis=0),
            'variance': np.var(self.data, axis=0),
            'zero_crossings': self._count_zero_crossings(),
            'stationarity_p_values': self._test_stationarity()
        }
       
    def _compute_snr(self):
        """Compute Signal-to-Noise Ratio."""
        signal_power = np.mean(self.data**2, axis=0)
        noise_power = np.var(np.diff(self.data, axis=0), axis=0)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
   
    def _count_zero_crossings(self):
        """Count zero crossings per channel."""
        return np.sum(np.diff(np.signbit(self.data), axis=0), axis=0)
   
    def _test_stationarity(self):
        """Test stationarity using Augmented Dickey-Fuller test."""
        from statsmodels.tsa.stattools import adfuller
        p_values = []
        for ch in range(self.data.shape[1]):
            try:
                result = adfuller(self.data[:, ch])
                p_values.append(result[1])
            except:
                p_values.append(1.0)  # Assume non-stationary if test fails
        return p_values
   
    def get_channel_data(self, channel: str) -> np.ndarray:
        """Get data for specific channel."""
        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not found")
        idx = self.channels.index(channel)
        return self.data[:, idx]
   
    def save_to_hdf5(self, filepath: str):
        """Save data to HDF5 format for efficient storage."""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('data', data=self.data)
            f.create_dataset('timestamps', data=self.timestamps)
            f.create_dataset('channels', data=[ch.encode() for ch in self.channels])
            f.attrs['fs'] = self.fs
            f.attrs['metadata'] = json.dumps(self.metadata)

class EnhancedEEGProcessor:
    """Enhanced EEG processing pipeline with advanced features."""
   
    def __init__(self, config: EEGConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
       
    def setup_logging(self):
        """Setup comprehensive logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = os.path.join(self.config.output_dir,
                               f"eeg_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
       
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
       
    def setup_directories(self):
        """Create necessary directories."""
        subdirs = ['plots', 'features', 'models', 'cache', 'intermediate']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.config.output_dir, subdir), exist_ok=True)
   
    def load_eeg_data(self, file_path: str) -> EEGData:
        """Enhanced data loading with validation and caching."""
        cache_file = self._get_cache_filename(file_path, 'raw_data')
       
        if os.path.exists(cache_file) and self.config.save_intermediate:
            self.logger.info(f"Loading cached data from {cache_file}")
            return self._load_from_cache(cache_file)
       
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded EEG data with {len(df)} records")
           
            # Filter for EEG Data rows
            eeg_df = df[df['Data Type'] == 'EEG Data'].copy()
           
            if eeg_df.empty:
                raise ValueError("No EEG data found in file")
           
            # Parse and validate data
            eeg_df = self._parse_eeg_channels(eeg_df)
           
            # Quality filtering
            if self.config.use_quality_check:
                eeg_df = self._apply_quality_filtering(df, eeg_df)
           
            # Convert to structured format
            eeg_data = self._convert_to_eeg_data(eeg_df)
           
            # Cache the processed data
            if self.config.save_intermediate:
                self._save_to_cache(eeg_data, cache_file)
           
            return eeg_data
           
        except Exception as e:
            self.logger.error(f"Failed to load EEG data: {str(e)}")
            raise
   
    def _parse_eeg_channels(self, eeg_df: pd.DataFrame) -> pd.DataFrame:
        """Parse EEG channel data with enhanced validation."""
        def parse_tuple_list(tuple_str):
            try:
                if pd.isna(tuple_str):
                    return np.array([])
               
                tuple_list = ast.literal_eval(tuple_str)
                valid_tuples = []
               
                for t in tuple_list:
                    if len(t) == 3:
                        try:
                            values = [float(x) for x in t]
                            # Validate physiological range for EEG (typically -200 to +200 μV)
                            if all(-500 <= v <= 500 for v in values):
                                valid_tuples.append(values)
                        except (ValueError, TypeError):
                            continue
               
                return np.array(valid_tuples) if valid_tuples else np.array([])
            except (ValueError, SyntaxError):
                return np.array([])
       
        eeg_df['F1'] = eeg_df['Left Channel'].apply(parse_tuple_list)
        eeg_df['F2'] = eeg_df['Right Channel'].apply(parse_tuple_list)
       
        # Remove invalid entries
        valid_mask = (eeg_df['F1'].apply(len) > 0) & (eeg_df['F2'].apply(len) > 0)
        eeg_df = eeg_df[valid_mask]
       
        self.logger.info(f"Valid EEG records after parsing: {len(eeg_df)}")
        return eeg_df
   
    def _apply_quality_filtering(self, df: pd.DataFrame, eeg_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filtering based on wear status and physiological signals."""
        try:
            # Check wear status
            wear_data = df[df['Data Type'] == 'Wear Status']
            if not wear_data.empty:
                valid_wear_times = wear_data[
                    wear_data['Left Channel'].astype(str).str.strip().str.lower() == 'true'
                ]['Timestamp'].values
               
                # Filter EEG data to valid wear times
                eeg_df = eeg_df[eeg_df['Timestamp'].isin(valid_wear_times)]
           
            # Additional quality checks can be added here
            self.logger.info(f"EEG records after quality filtering: {len(eeg_df)}")
            return eeg_df
           
        except Exception as e:
            self.logger.warning(f"Quality filtering failed: {str(e)}. Using all data.")
            return eeg_df
   
    def _convert_to_eeg_data(self, eeg_df: pd.DataFrame) -> EEGData:
        """Convert DataFrame to EEGData structure."""
        all_samples = []
        timestamps = []
       
        for idx, row in eeg_df.iterrows():
            f1 = row['F1']
            f2 = row['F2']
           
            if len(f1) == len(f2) and len(f1) > 0:
                # Combine F1 and F2 data
                combined_samples = np.hstack([f1, f2])  # Shape: (n_samples, 6)
                all_samples.extend(combined_samples)
               
                # Create timestamps for each sample
                base_time = pd.to_datetime(row['Timestamp']).timestamp()
                sample_times = base_time + np.arange(len(f1)) / self.config.fs
                timestamps.extend(sample_times)
       
        if not all_samples:
            raise ValueError("No valid EEG samples found")
       
        data = np.array(all_samples)
        timestamps = np.array(timestamps)
        channels = ['F1_0', 'F1_1', 'F1_2', 'F2_0', 'F2_1', 'F2_2']
       
        metadata = {
            'original_file': str(eeg_df.index[0]) if not eeg_df.empty else 'unknown',
            'processing_date': datetime.now().isoformat(),
            'n_original_records': len(eeg_df)
        }
       
        eeg_data = EEGData(data, timestamps, channels, self.config.fs, metadata)
        eeg_data.compute_quality_metrics()
       
        self.logger.info(f"Created EEG data: {data.shape[0]} samples, {data.shape[1]} channels")
        return eeg_data
   
    def apply_preprocessing(self, eeg_data: EEGData) -> EEGData:
        """Apply comprehensive preprocessing pipeline."""
        cache_file = self._get_cache_filename(eeg_data.metadata.get('original_file', 'unknown'), 'preprocessed')
       
        if os.path.exists(cache_file) and self.config.save_intermediate:
            self.logger.info("Loading cached preprocessed data")
            return self._load_from_cache(cache_file)
       
        data = eeg_data.data.copy()
       
        # Step 1: Basic filtering
        self.logger.info("Applying basic filtering...")
        data = self._apply_filters(data)
       
        # Step 2: Common Average Reference (CAR)
        if self.config.use_car:
            self.logger.info("Applying Common Average Reference...")
            data = self._apply_car(data)
       
        # Step 3: Advanced artifact removal
        self.logger.info("Removing artifacts...")
        data, artifacts = self._remove_artifacts_advanced(data)
       
        # Step 4: ICA for artifact removal
        if self.config.use_ica:
            self.logger.info("Applying ICA...")
            data, ica_info = self._apply_ica(data)
            artifacts['ica'] = ica_info
       
        # Step 5: Normalization
        self.logger.info("Normalizing data...")
        data = self._normalize_data(data)
       
        # Create new EEGData object
        processed_data = EEGData(
            data, eeg_data.timestamps, eeg_data.channels,
            eeg_data.fs, eeg_data.metadata
        )
        processed_data.artifacts = artifacts
        processed_data.compute_quality_metrics()
       
        # Cache the processed data
        if self.config.save_intermediate:
            self._save_to_cache(processed_data, cache_file)
       
        return processed_data
   
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters."""
        try:
            # Bandpass filter
            nyq = 0.5 * self.config.fs
            low = self.config.lowcut / nyq
            high = self.config.highcut / nyq
            b, a = signal.butter(self.config.filter_order, [low, high], btype='band')
           
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
           
            # Notch filter for power line interference
            b, a = signal.iirnotch(self.config.notch_freq, self.config.notch_quality, self.config.fs)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, filtered_data[:, i])
           
            return filtered_data
        except Exception as e:
            self.logger.error(f"Filtering failed: {str(e)}")
            raise
   
    def _apply_car(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference."""
        avg_ref = np.mean(data, axis=1, keepdims=True)
        return data - avg_ref
   
    def _remove_artifacts_advanced(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Advanced artifact removal using multiple methods."""
        artifacts = {}
        cleaned_data = data.copy()
       
        # Method 1: Statistical outlier removal
        z_scores = np.abs(stats.zscore(data, axis=0))
        outliers = z_scores > self.config.z_threshold
       
        for i in range(data.shape[1]):
            outlier_indices = outliers[:, i]
            if np.any(outlier_indices):
                # Use median interpolation for outliers
                median_val = np.median(data[~outlier_indices, i])
                cleaned_data[outlier_indices, i] = median_val
       
        artifacts['statistical_outliers'] = np.sum(outliers, axis=0)
       
        # Method 2: Gradient-based artifact detection
        gradients = np.abs(np.diff(data, axis=0))
        gradient_threshold = np.percentile(gradients, 99, axis=0)
       
        for i in range(data.shape[1]):
            steep_changes = gradients[:, i] > gradient_threshold[i]
            if np.any(steep_changes):
                # Smooth steep changes
                for idx in np.where(steep_changes)[0]:
                    if idx > 0 and idx < len(cleaned_data) - 2:  # Fixed boundary check
                        cleaned_data[idx+1, i] = (cleaned_data[idx, i] + cleaned_data[idx+2, i]) / 2
       
        artifacts['gradient_artifacts'] = np.sum(gradients > gradient_threshold, axis=0)
       
        self.logger.info(f"Removed artifacts: {artifacts}")
        return cleaned_data, artifacts
   
    def _apply_ica(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply Independent Component Analysis for artifact removal."""
        try:
            # Ensure we don't use more components than channels
            n_components = min(self.config.ica_components, data.shape[1])
           
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            components = ica.fit_transform(data)
           
            # Identify artifact components (simplified heuristic)
            artifact_components = []
            for i, comp in enumerate(components.T):
                # High variance components might be artifacts
                if np.var(comp) > 3 * np.mean([np.var(components[:, j]) for j in range(components.shape[1])]):
                    artifact_components.append(i)
           
            # Remove artifact components
            if artifact_components:
                components_clean = components.copy()
                components_clean[:, artifact_components] = 0
                cleaned_data = ica.inverse_transform(components_clean)
            else:
                cleaned_data = data
           
            ica_info = {
                'n_components': n_components,
                'artifact_components': artifact_components,
                'mixing_matrix': ica.mixing_,
                'components_variance': [np.var(components[:, i]) for i in range(components.shape[1])]
            }
           
            return cleaned_data, ica_info
           
        except Exception as e:
            self.logger.warning(f"ICA failed: {str(e)}. Skipping ICA.")
            return data, {}
   
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using robust scaling."""
        return self.robust_scaler.fit_transform(data)
   
    def segment_data(self, eeg_data: EEGData) -> List[EEGData]:
        """Segment data into overlapping windows."""
        segment_samples = int(self.config.segment_length * self.config.fs)
        overlap_samples = int(self.config.overlap * segment_samples)
        step_size = segment_samples - overlap_samples
       
        segments = []
        start_idx = 0
       
        while start_idx + segment_samples <= len(eeg_data.data):
            end_idx = start_idx + segment_samples
           
            segment_data = eeg_data.data[start_idx:end_idx]
            segment_timestamps = eeg_data.timestamps[start_idx:end_idx]
           
            # Create metadata for this segment
            segment_metadata = eeg_data.metadata.copy()
            segment_metadata.update({
                'segment_start_idx': start_idx,
                'segment_end_idx': end_idx,
                'segment_duration': self.config.segment_length
            })
           
            segment = EEGData(
                segment_data, segment_timestamps, eeg_data.channels,
                eeg_data.fs, segment_metadata
            )
            segments.append(segment)
           
            start_idx += step_size
       
        self.logger.info(f"Created {len(segments)} segments with {self.config.overlap*100}% overlap")
        return segments
   
    def _infer_cognitive_state(self, features: Dict) -> str:
        """Infer cognitive state based on EEG features using a simple heuristic."""
        try:
            # Example heuristic based on alpha and beta power for channel F1_0
            alpha_power = features.get('F1_0_alpha_power', 0)
            beta_power = features.get('F1_0_beta_power', 0)
            
            # Normalize powers for comparison (relative to total power)
            total_power = sum([features.get(f'F1_0_{band}_power', 0) 
                             for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']])
            alpha_ratio = alpha_power / total_power if total_power > 0 else 0
            beta_ratio = beta_power / total_power if total_power > 0 else 0
            
            # Heuristic rules
            if alpha_ratio > 0.3 and beta_ratio < 0.2:
                return "Relaxed"
            elif beta_ratio > 0.3 and alpha_ratio < 0.2:
                return "Focused"
            elif beta_ratio > 0.4:
                return "Stressed"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.warning(f"Failed to infer cognitive state: {str(e)}")
            return "Unknown"

    def extract_features(self, segments: List[EEGData]) -> pd.DataFrame:
        """Extract comprehensive features from all segments and infer cognitive state."""
        if self.config.n_jobs == 1:
            # Sequential processing
            all_features = [self._extract_segment_features(segment) for segment in tqdm(segments, desc="Extracting features")]
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.config.n_jobs if self.config.n_jobs > 0 else None) as executor:
                all_features = list(tqdm(
                    executor.map(self._extract_segment_features, segments),
                    total=len(segments),
                    desc="Extracting features"
                ))
       
        features_df = pd.DataFrame(all_features)
       
        # Add segment metadata and cognitive state
        for i, segment in enumerate(segments):
            features_df.loc[i, 'segment_id'] = i
            features_df.loc[i, 'timestamp'] = segment.timestamps[0]
            # Infer cognitive state for this segment
            features_df.loc[i, 'cognitive_state'] = self._infer_cognitive_state(all_features[i])
       
        self.logger.info(f"Extracted {len(features_df.columns)} features from {len(segments)} segments")
        return features_df
   
    def _extract_segment_features(self, segment: EEGData) -> Dict:
        """Extract comprehensive features from a single segment."""
        features = {}
        data = segment.data
        fs = segment.fs
       
        for ch_idx, channel in enumerate(segment.channels):
            ch_data = data[:, ch_idx]
           
            # Time-domain features
            features.update(self._extract_time_features(ch_data, channel))
           
            # Frequency-domain features
            features.update(self._extract_frequency_features(ch_data, channel, fs))
           
            # Nonlinear features
            features.update(self._extract_nonlinear_features(ch_data, channel))
           
            # Connectivity features (if multiple channels)
            if ch_idx < len(segment.channels) - 1:
                features.update(self._extract_connectivity_features(
                    ch_data, data[:, ch_idx + 1], f"{channel}_{segment.channels[ch_idx + 1]}"
                ))
       
        return features
   
    def _extract_time_features(self, data: np.ndarray, channel: str) -> Dict:
        """Extract time-domain features."""
        features = {}
       
        # Basic statistics
        features[f'{channel}_mean'] = np.mean(data)
        features[f'{channel}_std'] = np.std(data)
        features[f'{channel}_var'] = np.var(data)
        features[f'{channel}_skewness'] = stats.skew(data)
        features[f'{channel}_kurtosis'] = stats.kurtosis(data)
        features[f'{channel}_rms'] = np.sqrt(np.mean(data**2))
       
        # Range and percentiles
        features[f'{channel}_range'] = np.ptp(data)
        features[f'{channel}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{channel}_median'] = np.median(data)
       
        # Hjorth parameters
        hjorth = self._compute_hjorth_parameters(data)
        features[f'{channel}_hjorth_activity'] = hjorth[0]
        features[f'{channel}_hjorth_mobility'] = hjorth[1]
        features[f'{channel}_hjorth_complexity'] = hjorth[2]
       
        # Zero crossings
        features[f'{channel}_zero_crossings'] = len(np.where(np.diff(np.signbit(data)))[0])
       
        return features
   
    def _extract_frequency_features(self, data: np.ndarray, channel: str, fs: int) -> Dict:
        """Extract frequency-domain features."""
        features = {}
       
        # Power Spectral Density
        freqs, psd = signal.welch(data, fs=fs, nperseg=min(len(data), fs*2))
       
        # Band power features
        total_power = np.trapz(psd, freqs)
       
        for band_name, (low, high) in self.config.freq_bands.items():
            band_idx = (freqs >= low) & (freqs <= high)
            if np.any(band_idx):
                band_power = np.trapz(psd[band_idx], freqs[band_idx])
                features[f'{channel}_{band_name}_power'] = band_power
                features[f'{channel}_{band_name}_relative_power'] = band_power / total_power if total_power > 0 else 0
               
                # Peak frequency in band
                peak_idx = np.argmax(psd[band_idx])
                features[f'{channel}_{band_name}_peak_freq'] = freqs[band_idx][peak_idx]
       
        # Spectral features
        features[f'{channel}_spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features[f'{channel}_spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features[f'{channel}_spectral_centroid']) ** 2) * psd) / np.sum(psd))
        features[f'{channel}_spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
       
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        features[f'{channel}_spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
       
        return features
   
    def _extract_nonlinear_features(self, data: np.ndarray, channel: str) -> Dict:
        """Extract nonlinear features."""
        features = {}
       
        try:
            # Entropy measures
            features[f'{channel}_sample_entropy'] = ant.sample_entropy(data)
            features[f'{channel}_approximate_entropy'] = ant.app_entropy(data)
            features[f'{channel}_permutation_entropy'] = ant.perm_entropy(data)
           
            # Fractal dimension
            features[f'{channel}_higuchi_fd'] = ant.higuchi_fd(data)
            features[f'{channel}_katz_fd'] = ant.katz_fd(data)
           
            # Detrended Fluctuation Analysis
            features[f'{channel}_dfa'] = ant.detrended_fluctuation(data)
           
        except Exception as e:
            # If antropy fails, use basic entropy
            features[f'{channel}_entropy'] = entropy(np.histogram(data, bins=50)[0])
       
        return features
   
    def _extract_connectivity_features(self, data1: np.ndarray, data2: np.ndarray, channel_pair: str) -> Dict:
        """Extract connectivity features between channel pairs."""
        features = {}
       
        # Correlation
        features[f'{channel_pair}_correlation'] = np.corrcoef(data1, data2)[0, 1]
       
        # Coherence
        try:
            freqs, coherence = signal.coherence(data1, data2, fs=self.config.fs)
            for band_name, (low, high) in self.config.freq_bands.items():
                band_idx = (freqs >= low) & (freqs <= high)
                if np.any(band_idx):
                    features[f'{channel_pair}_{band_name}_coherence'] = np.mean(coherence[band_idx])
        except:
            pass
       
        # Phase synchronization
        try:
            analytic1 = hilbert(data1)
            analytic2 = hilbert(data2)
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
            phase_diff = phase1 - phase2
            features[f'{channel_pair}_phase_sync'] = np.abs(np.mean(np.exp(1j * phase_diff)))
        except:
            pass
       
        return features
   
    def _compute_hjorth_parameters(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Compute Hjorth parameters (Activity, Mobility, Complexity)."""
        # Activity
        activity = np.var(data)
       
        # Mobility
        first_deriv = np.diff(data)
        mobility = np.sqrt(np.var(first_deriv) / activity) if activity > 0 else 0
       
        # Complexity
        second_deriv = np.diff(first_deriv)
        complexity_mobility = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) if np.var(first_deriv) > 0 else 0
        complexity = complexity_mobility / mobility if mobility > 0 else 0
       
        return activity, mobility, complexity
   
    def create_visualizations(self, eeg_data: EEGData, segments: List[EEGData], features_df: pd.DataFrame):
        """Create comprehensive visualizations including cognitive state over time."""
        self.logger.info("Creating visualizations...")
       
        # Existing visualizations
        self._plot_raw_data(eeg_data)
        self._plot_quality_metrics(eeg_data)
        self._plot_psd_comparison(segments)
        self._plot_spectrograms(eeg_data)
        self._plot_feature_distributions(features_df)
        self._plot_feature_correlations(features_df)
        self._create_interactive_plots(eeg_data, features_df)
        
        # New: Cognitive state time-series plot
        self._plot_cognitive_state_timeseries(features_df)
       
        self.logger.info("Visualizations complete")
   
    def _plot_raw_data(self, eeg_data: EEGData):
        """Plot raw EEG data."""
        fig, axes = plt.subplots(len(eeg_data.channels), 1, figsize=(15, 2*len(eeg_data.channels)), sharex=True)
        if len(eeg_data.channels) == 1:
            axes = [axes]
       
        time_axis = np.arange(len(eeg_data.data)) / eeg_data.fs
       
        for i, (ax, channel) in enumerate(zip(axes, eeg_data.channels)):
            ax.plot(time_axis, eeg_data.data[:, i], linewidth=0.5)
            ax.set_ylabel(f'{channel}\n(μV)')
            ax.grid(True, alpha=0.3)
           
            # Add quality metrics as text
            if hasattr(eeg_data, 'quality_metrics') and eeg_data.quality_metrics:
                snr = eeg_data.quality_metrics.get('snr', [0])[i] if i < len(eeg_data.quality_metrics.get('snr', [])) else 0
                ax.text(0.02, 0.95, f'SNR: {snr:.1f} dB', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
       
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Raw EEG Data')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'raw_data.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_quality_metrics(self, eeg_data: EEGData):
        """Plot data quality metrics."""
        if not hasattr(eeg_data, 'quality_metrics') or not eeg_data.quality_metrics:
            return
       
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
       
        # SNR
        if 'snr' in eeg_data.quality_metrics:
            axes[0, 0].bar(eeg_data.channels, eeg_data.quality_metrics['snr'])
            axes[0, 0].set_title('Signal-to-Noise Ratio')
            axes[0, 0].set_ylabel('SNR (dB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
       
        # Variance
        if 'variance' in eeg_data.quality_metrics:
            axes[0, 1].bar(eeg_data.channels, eeg_data.quality_metrics['variance'])
            axes[0, 1].set_title('Signal Variance')
            axes[0, 1].set_ylabel('Variance')
            axes[0, 1].tick_params(axis='x', rotation=45)
       
        # Zero crossings
        if 'zero_crossings' in eeg_data.quality_metrics:
            axes[1, 0].bar(eeg_data.channels, eeg_data.quality_metrics['zero_crossings'])
            axes[1, 0].set_title('Zero Crossings')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
       
        # Stationarity p-values
        if 'stationarity_p_values' in eeg_data.quality_metrics:
            p_values = eeg_data.quality_metrics['stationarity_p_values']
            colors = ['red' if p > 0.05 else 'green' for p in p_values]
            axes[1, 1].bar(eeg_data.channels, p_values, color=colors)
            axes[1, 1].axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Stationarity Test (ADF)')
            axes[1, 1].set_ylabel('p-value')
            axes[1, 1].tick_params(axis='x', rotation=45)
       
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'quality_metrics.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_psd_comparison(self, segments: List[EEGData]):
        """Plot power spectral density comparison."""
        if not segments:
            return
       
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
       
        for ch_idx, channel in enumerate(segments[0].channels):
            ax = axes[ch_idx]
           
            # Plot PSD for first few segments
            for seg_idx, segment in enumerate(segments[:5]):
                freqs, psd = signal.welch(segment.data[:, ch_idx], fs=segment.fs)
                ax.semilogy(freqs, psd, alpha=0.7, label=f'Segment {seg_idx}')
           
            # Highlight frequency bands
            for band_name, (low, high) in self.config.freq_bands.items():
                ax.axvspan(low, high, alpha=0.1, label=band_name)
           
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD (μV²/Hz)')
            ax.set_title(f'Power Spectral Density - {channel}')
            ax.grid(True, alpha=0.3)
            ax.legend()
       
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'psd_comparison.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_spectrograms(self, eeg_data: EEGData):
        """Plot time-frequency representations (spectrograms)."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
       
        for ch_idx, channel in enumerate(eeg_data.channels):
            ax = axes[ch_idx]
           
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(eeg_data.data[:, ch_idx],
                                         fs=eeg_data.fs,
                                         nperseg=eeg_data.fs*2)
           
            # Plot spectrogram
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Spectrogram - {channel}')
            ax.set_ylim([0, 50])  # Focus on EEG frequencies
           
            plt.colorbar(im, ax=ax, label='Power (dB)')
       
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'spectrograms.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_feature_distributions(self, features_df: pd.DataFrame):
        """Plot feature distributions."""
        # Select key features for visualization
        key_features = [col for col in features_df.columns if any(x in col for x in ['mean', 'alpha_power', 'beta_power', 'sample_entropy'])]
       
        if not key_features:
            return
       
        n_features = min(len(key_features), 12)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
       
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
       
        for i, feature in enumerate(key_features[:n_features]):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
           
            # Histogram with KDE
            data = features_df[feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, density=True)
               
                # Add KDE if scipy available
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
                except:
                    pass
               
                ax.set_title(feature.replace('_', ' ').title())
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
       
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
       
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'feature_distributions.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_feature_correlations(self, features_df: pd.DataFrame):
        """Plot feature correlation matrix."""
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number])
       
        if numeric_features.empty:
            return
       
        # Compute correlation matrix
        corr_matrix = numeric_features.corr()
       
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, center=0,
                   square=True, cmap='RdBu_r',
                   cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'feature_correlations.{self.config.plot_format}'), dpi=300)
        plt.close()
   
    def _plot_cognitive_state_timeseries(self, features_df: pd.DataFrame):
        """Plot cognitive state over time using Plotly."""
        if 'cognitive_state' not in features_df.columns or 'timestamp' not in features_df.columns:
            self.logger.warning("Cognitive state or timestamp data missing for time-series plot")
            return
        
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(features_df['timestamp'], unit='s')
        states = features_df['cognitive_state']
        
        # Create a numeric encoding for states for plotting
        state_mapping = {'Relaxed': 1, 'Focused': 2, 'Stressed': 3, 'Neutral': 0, 'Unknown': -1}
        state_values = [state_mapping.get(state, -1) for state in states]
        
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=state_values,
                mode='lines+markers',
                name='Cognitive State',
                text=states,
                hovertemplate='Time: %{x}<br>State: %{text}',
                line=dict(shape='hv')  # Step-wise interpolation
            )
        )
        
        # Customize layout
        fig.update_layout(
            title='Cognitive State Over Time',
            xaxis_title='Time',
            yaxis_title='Cognitive State',
            yaxis=dict(
                tickvals=[-1, 0, 1, 2, 3],
                ticktext=['Unknown', 'Neutral', 'Relaxed', 'Focused', 'Stressed']
            ),
            showlegend=True,
            height=600
        )
        
        # Save as HTML
        output_path = os.path.join(self.config.output_dir, 'plots', 'cognitive_state_timeseries.html')
        fig.write_html(output_path)
        self.logger.info(f"Saved cognitive state time-series plot to {output_path}")
   
    def _create_interactive_plots(self, eeg_data: EEGData, features_df: pd.DataFrame):
        """Create interactive plots using Plotly."""
        # Interactive raw data plot
        fig = make_subplots(rows=len(eeg_data.channels), cols=1,
                           shared_xaxes=True,
                           subplot_titles=eeg_data.channels)
       
        time_axis = np.arange(len(eeg_data.data)) / eeg_data.fs
       
        for i, channel in enumerate(eeg_data.channels):
            fig.add_trace(
                go.Scatter(x=time_axis, y=eeg_data.data[:, i],
                          mode='lines', name=channel, line=dict(width=1)),
                row=i+1, col=1
            )
       
        fig.update_layout(height=200*len(eeg_data.channels),
                         title_text="Interactive EEG Data",
                         showlegend=False)
        fig.update_xaxes(title_text="Time (s)", row=len(eeg_data.channels), col=1)
       
        fig.write_html(os.path.join(self.config.output_dir, 'plots', 'interactive_raw_data.html'))
       
        # Interactive feature plot
        if not features_df.empty:
            numeric_features = features_df.select_dtypes(include=[np.number])
            if len(numeric_features.columns) >= 2:
                # Create scatter plot of first two principal components
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_features = pca.fit_transform(numeric_features.fillna(0))
                   
                    fig = px.scatter(x=pca_features[:, 0], y=pca_features[:, 1],
                                   title='Feature Space (PCA)')
                    fig.update_xaxes(title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    fig.update_yaxes(title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                   
                    fig.write_html(os.path.join(self.config.output_dir, 'plots', 'interactive_features.html'))
                except:
                    pass
   
    def train_classifier(self, features_df: pd.DataFrame, labels: np.ndarray = None) -> Dict:
        """Train machine learning classifier on extracted features."""
        if labels is None:
            # Create dummy labels for demonstration (e.g., based on time)
            labels = (np.arange(len(features_df)) > len(features_df) // 2).astype(int)
        
        # Prepare features data
        numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_features.empty:
            self.logger.warning("No numeric features available for classification")
            return {}
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )
       
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
       
        # Evaluate
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
       
        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, labels, cv=5)
       
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_features.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
       
        # Save model
        model_path = os.path.join(self.config.output_dir, 'models', 'classifier.joblib')
        joblib.dump(clf, model_path)
       
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', f'feature_importance.{self.config.plot_format}'), dpi=300)
        plt.close()
       
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'feature_importance': feature_importance,
            'model_path': model_path
        }
       
        self.logger.info(f"Classifier trained: Test accuracy = {test_score:.3f}")
        return results
   
    def generate_report(self, eeg_data: EEGData, segments: List[EEGData],
                       features_df: pd.DataFrame, ml_results: Dict = None) -> str:
        """Generate comprehensive analysis report including cognitive state analysis."""
        report_lines = []
        report_lines.append("# EEG Analysis Report\n")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
       
        # Data summary
        report_lines.append("## Data Summary")
        report_lines.append(f"- Total samples: {len(eeg_data.data):,}")
        report_lines.append(f"- Duration: {len(eeg_data.data) / eeg_data.fs:.1f} seconds")
        report_lines.append(f"- Sampling rate: {eeg_data.fs} Hz")
        report_lines.append(f"- Channels: {', '.join(eeg_data.channels)}")
        report_lines.append(f"- Segments created: {len(segments)}")
        report_lines.append("")
       
        # Data quality
        if hasattr(eeg_data, 'quality_metrics') and eeg_data.quality_metrics:
            report_lines.append("## Data Quality Metrics")
            for metric, values in eeg_data.quality_metrics.items():
                if isinstance(values, (list, np.ndarray)):
                    report_lines.append(f"- {metric.replace('_', ' ').title()}: {np.mean(values):.2f} ± {np.std(values):.2f}")
                else:
                    report_lines.append(f"- {metric.replace('_', ' ').title()}: {values}")
            report_lines.append("")
       
        # Artifacts
        if hasattr(eeg_data, 'artifacts') and eeg_data.artifacts:
            report_lines.append("## Artifact Removal")
            for artifact_type, info in eeg_data.artifacts.items():
                if isinstance(info, (list, np.ndarray)) and len(info) > 0:
                    report_lines.append(f"- {artifact_type.replace('_', ' ').title()}: {np.sum(info)} artifacts removed")
            report_lines.append("")
       
        # Feature summary
        if not features_df.empty:
            report_lines.append("## Feature Summary")
            numeric_features = features_df.select_dtypes(include=[np.number])
            report_lines.append(f"- Total features extracted: {len(numeric_features.columns)}")
            
            # Feature categories
            feature_categories = {}
            for col in numeric_features.columns:
                category = col.split('_')[-1] if '_' in col else 'other'
                feature_categories[category] = feature_categories.get(category, 0) + 1
            
            for category, count in feature_categories.items():
                report_lines.append(f"- {category.title()} features: {count}")
            report_lines.append("")
        
        # Cognitive state analysis
        if 'cognitive_state' in features_df.columns:
            report_lines.append("## Cognitive State Analysis")
            state_counts = features_df['cognitive_state'].value_counts()
            total_segments = len(features_df)
            
            report_lines.append(f"- Total segments analyzed: {total_segments}")
            for state, count in state_counts.items():
                percentage = (count / total_segments) * 100
                report_lines.append(f"- {state}: {count} segments ({percentage:.1f}%)")
            
            # Dominant state
            dominant_state = state_counts.idxmax() if not state_counts.empty else "Unknown"
            report_lines.append(f"- Dominant cognitive state: {dominant_state}")
            
            # Trend analysis (simple: check if state changes occur)
            state_changes = (features_df['cognitive_state'] != features_df['cognitive_state'].shift(1)).sum()
            report_lines.append(f"- Number of state transitions: {state_changes - 1 if state_changes > 0 else 0}")
            report_lines.append("")
       
        # Machine learning results
        if ml_results:
            report_lines.append("## Machine Learning Results")
            report_lines.append(f"- Training accuracy: {ml_results.get('train_accuracy', 0):.3f}")
            report_lines.append(f"- Test accuracy: {ml_results.get('test_accuracy', 0):.3f}")
            report_lines.append(f"- Cross-validation: {ml_results.get('cv_mean', 0):.3f} ± {ml_results.get('cv_std', 0):.3f}")
            
            if 'feature_importance' in ml_results:
                top_features = ml_results['feature_importance'].head(5)
                report_lines.append("\n### Top 5 Important Features:")
                for _, row in top_features.iterrows():
                    report_lines.append(f"- {row['feature']}: {row['importance']:.3f}")
            report_lines.append("")
       
        # Configuration
        report_lines.append("## Processing Configuration")
        config_dict = asdict(self.config)
        for key, value in config_dict.items():
            if not key.startswith('_'):
                report_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
       
        report_content = '\n'.join(report_lines)
       
        # Save report
        report_path = os.path.join(self.config.output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
       
        self.logger.info(f"Report saved to {report_path}")
        return report_content
   
    def _get_cache_filename(self, original_file: str, stage: str) -> str:
        """Generate cache filename based on file and processing stage."""
        file_hash = hashlib.md5(str(original_file).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(str(asdict(self.config)).encode()).hexdigest()[:8]
        return os.path.join(self.config.output_dir, 'cache', f"{stage}_{file_hash}_{config_hash}.pkl")
   
    def _save_to_cache(self, data: EEGData, cache_file: str):
        """Save EEGData to cache."""
        try:
            joblib.dump(data, cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {str(e)}")
   
    def _load_from_cache(self, cache_file: str) -> EEGData:
        """Load EEGData from cache."""
        return joblib.load(cache_file)
   
    def process_pipeline(self, file_path: str, labels: np.ndarray = None) -> Dict:
        """Run the complete EEG processing pipeline."""
        self.logger.info("Starting EEG processing pipeline...")
       
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading EEG data...")
            eeg_data = self.load_eeg_data(file_path)
           
            # Step 2: Preprocessing
            self.logger.info("Step 2: Applying preprocessing...")
            processed_data = self.apply_preprocessing(eeg_data)
           
            # Step 3: Segmentation
            self.logger.info("Step 3: Segmenting data...")
            segments = self.segment_data(processed_data)
           
            # Step 4: Feature extraction
            self.logger.info("Step 4: Extracting features...")
            features_df = self.extract_features(segments)
           
            # Step 5: Machine learning (optional)
            ml_results = {}
            if labels is not None:
                self.logger.info("Step 5: Training classifier...")
                ml_results = self.train_classifier(features_df, labels)
           
            # Step 6: Visualization
            self.logger.info("Step 6: Creating visualizations...")
            self.create_visualizations(processed_data, segments, features_df)
           
            # Step 7: Save results
            self.logger.info("Step 7: Saving results...")
            features_path = os.path.join(self.config.output_dir, 'features', 'eeg_features.csv')
            features_df.to_csv(features_path, index=False)
           
            # Save configuration
            config_path = os.path.join(self.config.output_dir, 'config.yaml')
            self.config.save_config(config_path)
           
            # Step 8: Generate report
            self.logger.info("Step 8: Generating report...")
            report = self.generate_report(processed_data, segments, features_df, ml_results)
           
            results = {
                'eeg_data': processed_data,
                'segments': segments,
                'features': features_df,
                'ml_results': ml_results,
                'report': report,
                'output_dir': self.config.output_dir
            }
           
            self.logger.info("Pipeline completed successfully!")
            return results
           
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

# Example usage and testing
def main():
    """Main function demonstrating the enhanced EEG processing pipeline."""
   
    # Create configuration
    config = EEGConfig(
        fs=256,
        segment_length=2.0,
        overlap=0.5,
        use_ica=True,
        use_car=True,
        output_dir="enhanced_eeg_analysis",
        n_jobs=4  # Use 4 cores for parallel processing
    )
   
    # Initialize processor
    processor = EnhancedEEGProcessor(config)
   
    # Example file path (replace with your actual file)
    file_path = r"C:\grad\headband\headband\PSYCHONOVA\Collected Data\Bakr Mohamed\EEG (game 1).csv"
   
    try:
        # Run the complete pipeline
        results = processor.process_pipeline(file_path)
       
        print("Processing completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        print(f"Features shape: {results['features'].shape}")
        print(f"Number of segments: {len(results['segments'])}")
       
        # Display some key results
        if results['ml_results']:
            print(f"ML Test Accuracy: {results['ml_results']['test_accuracy']:.3f}")
       
        print("\nFirst few feature columns:")
        print(results['features'].columns[:10].tolist())
       
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
