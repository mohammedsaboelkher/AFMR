import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi


def get_snr(y_orig: np.ndarray, y_watermarked: np.ndarray) -> float:
    """Calculates the Signal-to-Noise Ratio (SNR).

    The ratio of the original clean audio power to the power of the "noise"
    introduced by the watermark. Higher values indicate less distortion.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.

    Returns:
        float: The SNR in decibels (dB).
    """
    signal_power = np.sum(y_orig**2)
    noise_power = np.sum((y_orig - y_watermarked) ** 2)
    return float(10 * np.log10(signal_power / (noise_power + 1e-9)))


def get_log_spectral_distance(y_orig: np.ndarray, y_watermarked: np.ndarray) -> float:
    """Calculates the Log Spectral Distance (LSD).

    Measures the distance between the spectrograms of the original and watermarked
    audio. It quantifies how much the frequency distribution was shifted or damaged.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.

    Returns:
        float: The mean Log Spectral Distance.
    """
    S_orig = np.log10(np.abs(librosa.stft(y_orig)) + 1e-9)
    S_wm = np.log10(np.abs(librosa.stft(y_watermarked)) + 1e-9)
    return float(np.mean(np.sqrt(np.mean((S_orig - S_wm) ** 2, axis=0))))


def get_mel_cepstral_distortion(
    y_orig: np.ndarray, y_watermarked: np.ndarray, sr: int
) -> float:
    """Calculates the Mel Cepstral Distortion (MCD).

    An objective distance measure calculating how much the timbre or vocal tract
    shape of the audio was altered by the embedding process.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        float: The Mel Cepstral Distortion score.
    """
    mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr)
    mfcc_wm = librosa.feature.mfcc(y=y_watermarked, sr=sr)
    constant = 10.0 / np.log(10.0) * np.sqrt(2.0)
    diff = mfcc_orig - mfcc_wm
    return float(np.mean(constant * np.sqrt(np.sum(diff**2, axis=0))))


def get_stft_difference(y_orig: np.ndarray, y_watermarked: np.ndarray) -> float:
    """Calculates the mean absolute difference in the STFT domain.

    Provides a raw, linear mathematical difference between the Short-Time
    Fourier Transforms of both files.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.

    Returns:
        float: The mean STFT difference.
    """
    S_orig = np.abs(librosa.stft(y_orig))
    S_wm = np.abs(librosa.stft(y_watermarked))
    return float(np.mean(np.abs(S_orig - S_wm)))


def get_pesq_score(y_orig: np.ndarray, y_watermarked: np.ndarray, sr: int) -> float:
    """Calculates the Perceptual Evaluation of Speech Quality (PESQ) score.

    A complex, industry-standard algorithm that predicts the Mean Opinion Score
    (1.0 to 4.5) that a human listener would give the degraded audio.
    Note: PESQ standard requires audio at 8kHz or 16kHz. Audio will be resampled
    to 16kHz internally if it does not match.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        float: The PESQ score (typically between 1.0 and 4.5).
    """
    if sr not in [8000, 16000]:
        y_orig = librosa.resample(y_orig, orig_sr=sr, target_sr=16000)
        y_watermarked = librosa.resample(y_watermarked, orig_sr=sr, target_sr=16000)
        sr = 16000
    return float(pesq(sr, y_orig, y_watermarked, "wb"))


def get_stoi_score(y_orig: np.ndarray, y_watermarked: np.ndarray, sr: int) -> float:
    """Calculates the Short-Time Objective Intelligibility (STOI) score.

    Calculates a score (0.0 to 1.0) representing the proportion of the speech
    that can still be understood by a human listener after watermarking degradation.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        float: The STOI score (0.0 to 1.0).
    """
    return float(stoi(y_orig, y_watermarked, sr, extended=False))


def get_total_phase_error(y_orig: np.ndarray, y_watermarked: np.ndarray) -> float:
    """Calculates the mean absolute phase difference in the STFT domain.

    Extracts the phase angle spectrum of both the original and watermarked
    signals and measures the absolute angular deviation between them.

    Significance:
        If an embedding algorithm uses phase modulation to hide data, the
        magnitude metrics (like SNR or STFT difference) might show perfect 0
        distortion, while this phase error metric will light up completely.

    Args:
        y_orig (np.ndarray): Original audio time series.
        y_watermarked (np.ndarray): Modified/Watermarked audio time series.

    Returns:
        float: Mean absolute phase deviation in radians (0.0 to pi).
    """
    stft_orig = librosa.stft(y_orig)
    stft_wm = librosa.stft(y_watermarked)

    phase_orig = np.angle(stft_orig)
    phase_wm = np.angle(stft_wm)

    # Calculate angular distance, wrapping correctly around -pi and pi
    phase_diff = np.abs(
        np.arctan2(np.sin(phase_orig - phase_wm), np.cos(phase_orig - phase_wm))
    )
    return float(np.mean(phase_diff))
