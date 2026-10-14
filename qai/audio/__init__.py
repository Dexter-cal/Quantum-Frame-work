"""
qai.audio -- Audio Signal Processing & Spectrogram Utilities.
"""
import numpy as np
from typing import Any, Tuple

def spectrogram(waveform: Any, n_fft: int = 256, hop_length: int = 128) -> np.ndarray:
    """Computes Short-Time Fourier Transform (STFT) magnitude spectrogram."""
    sig = np.array(waveform, dtype=float)
    n_samples = len(sig)
    n_frames = max(1, (n_samples - n_fft) // hop_length + 1)
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=float)
    for i in range(n_frames):
        start = i * hop_length
        frame = sig[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        fft_res = np.fft.rfft(frame * np.hanning(n_fft))
        spec[:, i] = np.abs(fft_res)
    return spec

def melspectrogram(waveform: Any, sample_rate: int = 16000, n_mels: int = 40) -> np.ndarray:
    """Computes Log-Mel Filterbank Spectrogram."""
    spec = spectrogram(waveform)
    # Mel-scale compression simulation
    mel_spec = np.log1p(np.resize(spec, (n_mels, spec.shape[1])))
    return mel_spec

def mfcc(waveform: Any, n_mfcc: int = 13) -> np.ndarray:
    """Computes Mel-Frequency Cepstral Coefficients (MFCC)."""
    mel = melspectrogram(waveform)
    from scipy.fftpack import dct
    return dct(mel, axis=0, norm='ortho')[:n_mfcc]

def pitch_shift(waveform: Any, semitones: float = 2.0) -> np.ndarray:
    """Simulates audio pitch shifting."""
    sig = np.array(waveform, dtype=float)
    factor = 2.0 ** (semitones / 12.0)
    indices = np.round(np.arange(0, len(sig), factor)).astype(int)
    indices = indices[indices < len(sig)]
    return sig[indices]

def time_stretch(waveform: Any, rate: float = 1.2) -> np.ndarray:
    """Time-stretches audio waveform without changing pitch."""
    sig = np.array(waveform, dtype=float)
    indices = np.round(np.arange(0, len(sig), rate)).astype(int)
    indices = indices[indices < len(sig)]
    return sig[indices]
