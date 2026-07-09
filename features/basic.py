import librosa
import numpy as np


def get_duration(y: np.ndarray, sr: int) -> float:
    """Calculates the total length of the audio file.

    Duration is a fundamental macroscopic property. In watermarking, it is
    essential for calculating bitrates (bits embedded per second).

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The duration of the audio in seconds.
    """
    return librosa.get_duration(y=y, sr=sr)


def get_sample_rate(sr: int) -> int:
    """Returns the sample rate of the audio.

    The number of audio samples carried per second. Crucial for determining
    the Nyquist frequency (maximum representable frequency) of the file.

    Args:
        sr (int): Sampling rate of the audio.

    Returns:
        int: The sampling rate in Hz.
    """
    return sr


def get_peak_amplitude(y: np.ndarray) -> float:
    """Calculates the absolute highest volume point in the track.

    Peak amplitude is the maximum sample value. It is primarily used to
    check for clipping or digital distortion in the signal.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The maximum absolute amplitude value.
    """
    return float(np.max(np.abs(y)))


def get_rms_energy(y: np.ndarray) -> float:
    """Calculates the Root Mean Square (RMS) energy of the signal.

    RMS calculates the average power/energy of the signal over time. It
    provides a rough, mathematically straightforward estimate of perceived volume.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The mean RMS energy.
    """
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def get_dynamic_range(y: np.ndarray) -> float:
    """Calculates the dynamic range of the audio.

    Dynamic range is the ratio between the loudest possible sound and the
    quietest background noise in the signal. A high dynamic range means a
    wide variance between whispers and shouts.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The dynamic range in decibels (dB).
    """
    rms = librosa.feature.rms(y=y)[0]
    return float(20 * np.log10(np.max(rms) / (np.min(rms[rms > 0]) + 1e-9)))


def get_silence_ratio(y: np.ndarray, threshold_db: float = -60.0) -> float:
    """Calculates the percentage of the audio that is effectively silent.

    Useful for understanding the sparsity of speech. High silence ratios might
    limit where a watermark can be imperceptibly hidden.

    Args:
        y (np.ndarray): Audio time series.
        threshold_db (float, optional): The decibel limit below which audio
            is considered silent. Defaults to -60.0.

    Returns:
        float: The ratio of silent frames to total frames (0.0 to 1.0).
    """
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    silence_frames = np.sum(db < threshold_db)
    return float(silence_frames / len(db))


def get_zero_crossing_rate(y: np.ndarray) -> float:
    """Calculates how often the audio signal crosses the zero-amplitude line.

    High rates indicate "noisy" or "scratchy" sounds (like the letter "S"
    or high-hats). Low rates indicate tonal, lower-frequency sounds.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The mean zero-crossing rate.
    """
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(zcr))


def get_crest_factor(y: np.ndarray) -> float:
    """Calculates the Crest Factor of the waveform.

    The Crest Factor is the ratio of the peak amplitude to the RMS value of the
    waveform, expressed in decibels. It indicates how impulsive or "peaky" a
    signal is.

    Significance:
        Watermarks that inject high-frequency noise or dither often lower the
        Crest Factor. Conversely, brickwall limiting or heavy clipping noticeably
        crushes this metric, making it a great integrity check.

    Args:
        y (np.ndarray): Audio time series.

    Returns:
        float: The Crest Factor in decibels (dB).
    """
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return 0.0
    return float(20 * np.log10(peak / rms))
