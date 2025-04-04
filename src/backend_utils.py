import numpy as np
import pyloudnorm as pyln
import librosa


def get_peak_normalization(y: np.ndarray, target_peak: float = 0.95) -> np.array:
    max_val = np.max(np.abs(y))
    y_normalized = y * (target_peak / max_val)
    return y_normalized


def get_rms_normalization(y: np.ndarray, target_dB: float = -10) -> np.array:
    rms = np.sqrt(np.mean(y ** 2))
    target_rms = 10 ** (target_dB / 20)
    y_normalized = y * (target_rms / rms)
    return y_normalized


def get_lufs_normalization(y: np.ndarray, sr: int = 16000, target_dB: float = -10) -> np.array:
        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(y)
        y_normalized = pyln.normalize.loudness(y, current_loudness, target_dB)
        return y_normalized


def get_resampled_and_normalized_audio(
        audio: np.ndarray,
        norm_algo: str,
        target_peak: float,
        target_dB: float,
        model_sr: int,
        original_sr: int,
) -> np.ndarray:
    audio = librosa.resample(y=audio, orig_sr=original_sr, target_sr=model_sr)
    if norm_algo == "peak":
        audio = get_peak_normalization(audio, target_peak=target_peak)
    elif norm_algo == "rms":
        audio = get_rms_normalization(audio, target_dB=target_dB)
    elif norm_algo == "lufs":
        audio = get_lufs_normalization(audio, sr=model_sr, target_dB=target_dB)
    return audio
