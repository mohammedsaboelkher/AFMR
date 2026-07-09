import librosa
import pyloudnorm as pyln
import numpy as np
import scipy.signal as signal


def get_lufs(y: np.ndarray, sr: int) -> float:
    """Calculates integrated Loudness Units relative to Full Scale (LUFS).

    The broadcast standard for perceived loudness. It uses psychoacoustic filters
    that mimic human hearing, recognizing that humans hear mid-frequencies louder.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The integrated LUFS value.
    """
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))


def get_loudness(y: np.ndarray) -> float:
    """Calculates a simplified perceptual loudness proxy.

    While true loudness is complex (e.g., A-weighting, ISO 226), the mean RMS
    is a functional proxy for general signal energy if LUFS is unavailable.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The proxy loudness score (mean RMS).
    """
    return float(np.mean(librosa.feature.rms(y=y)))


def get_sharpness(y: np.ndarray, sr: int) -> float:
    """Calculates audio sharpness (approximated).

    Sharpness explicitly measures the sensation of "harshness" caused by
    excessive high-frequency energy. Approximated here via Spectral Centroid.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The approximated sharpness score.
    """
    return get_spectral_centroid(y, sr)


def get_roughness(y: np.ndarray, sr: int) -> float:
    """Calculates audio roughness (approximated).

    The subjective perception of rapid amplitude modulation (usually 15-300 Hz),
    which sounds "buzzing" or "grating". Approximated via amplitude modulation depth.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The approximated roughness score.
    """
    envelope = np.abs(signal.hilbert(y))
    return float(np.var(np.diff(envelope)))
