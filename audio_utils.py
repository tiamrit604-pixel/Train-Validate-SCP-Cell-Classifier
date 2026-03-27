"""
audio_utils.py
--------------
Exact replication of the feature extraction pipeline from the notebook.
All logic here matches HW3_Code_-_newdataset_-_Grid_CV_Final.ipynb
"""

import numpy as np
import librosa
import audioread
import tempfile
import os
from scipy.signal import find_peaks, periodogram


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_bytes(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes (from Streamlit uploader).
    Writes to a temp file, decodes with audioread, converts to float32.
    Supports .wav, .m4a, .mp3, .ogg, .flac
    """
    suffix = os.path.splitext(filename)[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Use librosa for broad format support (calls ffmpeg/soundfile internally)
        signal, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.unlink(tmp_path)

    return signal.astype(np.float32), sr


def load_audio_path(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio from a file path (used during training from disk)."""
    signal, sr = librosa.load(filepath, sr=None, mono=True)
    return signal.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Peak detection & segmentation  (exact notebook logic)
# ---------------------------------------------------------------------------

def detect_peaks_and_segment(signal: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Detects impulse peaks in signal and extracts segments.
    - height threshold: 0.3 * max amplitude
    - minimum distance: 0.5 seconds
    - pre-peak window: 20 ms
    - post-peak window: 200 ms
    - each segment normalized to [-1, 1]
    """
    peaks, _ = find_peaks(
        np.abs(signal),
        height=0.3 * np.max(np.abs(signal)),
        distance=int(0.5 * sr)
    )

    if len(peaks) == 0:
        # Fallback: treat whole signal center as one peak
        peaks = [len(signal) // 2]

    pre_samples  = int(0.02 * sr)
    post_samples = int(0.20 * sr)

    segments = []
    for peak in peaks:
        start   = max(0, peak - pre_samples)
        end     = min(len(signal), peak + post_samples)
        segment = signal[start:end].copy()

        max_val = np.max(np.abs(segment))
        if max_val > 0:
            segment = segment / max_val

        segments.append(segment)

    return segments


# ---------------------------------------------------------------------------
# Feature extraction  (exact notebook logic)
# ---------------------------------------------------------------------------

def extract_features(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts a 264-dimensional feature vector from one segment:
      - 200 log-PSD features  (periodogram, nfft=4096, first 200 bins)
      -  32 MFCC mean features (librosa, n_mfcc=32)
      -  32 MFCC std  features
    Total: 264 features
    """
    x = segment.astype(np.float64)

    # --- PSD ---
    _, pxx = periodogram(
        x,
        fs=1.0,
        window='boxcar',
        nfft=4096,
        detrend=False,
        scaling='density',
        return_onesided=True
    )
    psd_features = np.log(pxx[:200] + 1e-10)

    # --- MFCC ---
    mfcc_mat  = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)
    mfcc_mean = mfcc_mat.mean(axis=1)   # shape (32,)
    mfcc_std  = mfcc_mat.std(axis=1)    # shape (32,)

    feature_vector = np.concatenate([psd_features, mfcc_mean, mfcc_std])
    return feature_vector


# ---------------------------------------------------------------------------
# High-level: process a list of (bytes, filename) pairs -> X, y, meta
# ---------------------------------------------------------------------------

def process_uploaded_files(
    uploaded_files,          # list of st.UploadedFile objects
    has_labels: bool = True
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Process a list of Streamlit UploadedFile objects.

    Labeling rule (same as notebook):
      filename ends with 'g.<ext>'  →  Good (1)
      anything else                 →  Bad  (0)

    Returns
    -------
    X     : (N, 264) feature matrix
    y     : (N,) label array  (all -1 when has_labels=False)
    meta  : list of dicts with keys: filename, label, segment_idx, n_segments
    """
    X, y, meta = [], [], []

    for uf in uploaded_files:
        filename = uf.name
        file_bytes = uf.read()

        try:
            signal, sr = load_audio_bytes(file_bytes, filename)
        except Exception as e:
            # Return error info but skip
            meta.append({
                "filename": filename,
                "label": -1,
                "segment_idx": 0,
                "n_segments": 0,
                "error": str(e)
            })
            continue

        segments   = detect_peaks_and_segment(signal, sr)
        n_segments = len(segments)

        if has_labels:
            label = 1 if filename.strip().lower().endswith(("g.wav", "g.m4a",
                                                             "g.mp3", "g.ogg",
                                                             "g.flac")) else 0
        else:
            label = -1

        for seg_idx, seg in enumerate(segments):
            feat = extract_features(seg, sr)
            X.append(feat)
            y.append(label)
            meta.append({
                "filename": filename,
                "label": label,
                "segment_idx": seg_idx,
                "n_segments": n_segments,
                "error": None,
                "signal": signal if seg_idx == 0 else None,  # store raw for plotting
                "sr": sr
            })

    X = np.array(X) if X else np.empty((0, 264))
    y = np.array(y) if y else np.array([])
    return X, y, meta
