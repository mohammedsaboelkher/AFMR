import librosa
import numpy as np
import scipy.signal as signal


def get_speech_rate(y: np.ndarray, sr: int) -> float:
    """Approximates the speech rate via onset detection.

    Estimates the number of spoken units (usually syllable nuclei) per second.
    Crucial for capturing the speaker's cadence and tempo.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The estimated speech rate in syllables/onsets per second.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = signal.find_peaks(onset_env, prominence=1)
    duration = librosa.get_duration(y=y, sr=sr)
    return float(len(peaks) / duration) if duration > 0 else 0.0


def get_pause_statistics(
    y: np.ndarray, sr: int, top_db: float = 40.0
) -> tuple[float, float]:
    """Calculates the mean and variance of pause durations.

    Measures the average length and variance of silences between words
    or sentences. Dictates macro-level rhythm and pacing.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        top_db (float, optional): The threshold (in dB) below reference to consider
            as silence. Defaults to 40.0.

    Returns:
        tuple[float, float]: The (mean, variance) of pause durations in seconds.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) <= 1:
        return 0.0, 0.0
    pauses = []
    for i in range(1, len(intervals)):
        pause_samples = intervals[i][0] - intervals[i - 1][1]
        pauses.append(pause_samples / sr)
    return float(np.mean(pauses)), float(np.var(pauses))


def get_modulation_spectrum(y: np.ndarray, sr: int) -> float:
    """Calculates the mean modulation spectrum.

    Measures rhythm at a macro level by looking at how the overall volume
    envelope oscillates over time, capturing the frequency of loudness fluctuations.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The mean magnitude of the modulation spectrum.
    """
    envelope = np.abs(signal.hilbert(y))
    mod_spec = np.abs(np.fft.rfft(envelope))
    return float(np.mean(mod_spec))


def get_energy_envelope_variance(y: np.ndarray) -> float:
    """Calculates the variance of the signal's energy envelope.

    Tracks how drastically the overall volume swells and fades. High variance
    implies highly dynamic speech; low variance implies a monotone delivery.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The variance of the analytic signal's envelope.
    """
    envelope = np.abs(signal.hilbert(y))
    return float(np.var(envelope))
