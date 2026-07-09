import librosa
import numpy as np


def get_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """Calculates the spectral centroid, or the "center of mass" of the spectrum.

    Indicates how "bright" a sound is. A bass guitar has a low centroid;
    a cymbal has a high one. High centroids often correlate with perceived sharpness.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The mean spectral centroid in Hz.
    """
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(cent))


def get_spectral_bandwidth(y: np.ndarray, sr: int) -> float:
    """Calculates the spectral bandwidth around the centroid.

    It tells you if the sound occupies a narrow frequency range (like a whistle)
    or a wide one (like white noise).

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The mean spectral bandwidth in Hz.
    """
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    return float(np.mean(bw))


def get_spectral_rolloff(y: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    """Calculates the spectral roll-off frequency.

    The frequency below which a specified percentage of total spectral energy
    is contained. Helps distinguish between harmonic sounds and noisy sounds.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        roll_percent (float, optional): Percentage of energy. Defaults to 0.85.

    Returns:
        float: The mean roll-off frequency in Hz.
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    return float(np.mean(rolloff))


def get_spectral_flatness(y: np.ndarray) -> float:
    """Calculates how flat the frequency spectrum is.

    A high value (near 1) means the sound is noise-like (energy spread evenly);
    a low value (near 0) means it is tonal (energy concentrated in specific pitches).

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The mean spectral flatness.
    """
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    return float(np.mean(flatness))


def get_spectral_entropy(y: np.ndarray) -> float:
    """Calculates the Shannon entropy of the power spectrogram.

    Captures the "randomness" or unpredictability of the frequency spectrum.
    White noise has high entropy; a pure sine wave has near-zero entropy.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The mean spectral entropy.
    """
    S, _ = librosa.magphase(librosa.stft(y))
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    entropy = -np.sum(S_norm * np.log2(S_norm + 1e-9), axis=0)
    return float(np.mean(entropy))


def get_spectral_flux(y: np.ndarray) -> float:
    """Calculates the spectral flux of the audio.

    Measures how quickly the frequency spectrum is changing from one frame
    to the next. High flux usually happens at note onsets or transient strikes.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The mean spectral flux.
    """
    S, _ = librosa.magphase(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    return float(np.mean(flux))


def get_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    """Calculates the Spectral Contrast across frequency bands.

    Measures the energy difference between the peaks (harmonic components) and
    valleys (noise/background components) across distinct sub-bands of the spectrum.

    Significance:
        Many watermarking techniques hide data precisely inside the spectral valleys
        to ensure psychoacoustic masking. Tracking changes in Spectral Contrast
        instantly exposes if a watermark has altered the clear gaps between harmonics.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        np.ndarray: Mean spectral contrast value for each sub-band.
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(contrast, axis=1)
