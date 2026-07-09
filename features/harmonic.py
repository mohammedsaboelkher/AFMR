import librosa
import parselmouth
import numpy as np


def get_f0_mean_variance(audio_path: str) -> tuple[float, float]:
    """Calculates the mean and variance of the Fundamental Frequency (F0).

    F0 Mean is the average pitch of the voice. F0 Variance measures how much
    the pitch fluctuates (intonation and melody). Extracts using Praat for clinical accuracy.

    Args:
        audio_path (str): Filepath to the audio file.

    Returns:
        tuple[float, float]: A tuple containing the (mean, variance) of F0 in Hz.
    """
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
    if len(pitch_values) == 0:
        return 0.0, 0.0
    return float(np.mean(pitch_values)), float(np.var(pitch_values))


def get_jitter(audio_path: str) -> float:
    """Calculates local Jitter using Praat.

    Jitter measures micro-instabilities in *frequency* from cycle to cycle
    of the vocal cords. High jitter sounds "hoarse" or "creaky".

    Args:
        audio_path (str): Filepath to the audio file.

    Returns:
        float: The local jitter value (as a percentage/ratio).
    """
    snd = parselmouth.Sound(audio_path)
    pointProcess = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )
    return float(
        parselmouth.praat.call(
            pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
    )


def get_shimmer(audio_path: str) -> float:
    """Calculates local Shimmer using Praat.

    Shimmer measures micro-instabilities in *amplitude* (volume) from cycle
    to cycle. High shimmer sounds "breathy" or "rough".

    Args:
        audio_path (str): Filepath to the audio file.

    Returns:
        float: The local shimmer value (as a percentage/ratio).
    """
    snd = parselmouth.Sound(audio_path)
    pointProcess = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )
    return float(
        parselmouth.praat.call(
            [snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    )


def get_hnr(audio_path: str) -> float:
    """Calculates the Harmonics-to-Noise Ratio (HNR).

    The ratio of periodic acoustic energy (the actual voice) to aperiodic
    noise (breath, static, rasp). Higher HNR denotes a clearer voice.

    Args:
        audio_path (str): Filepath to the audio file.

    Returns:
        float: The mean HNR in decibels (dB).
    """
    snd = parselmouth.Sound(audio_path)
    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    return float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))


def get_voiced_ratio(y: np.ndarray, sr: int) -> float:
    """Calculates the ratio of voiced frames to total frames.

    Determines the percentage of the speech signal where the vocal cords are
    vibrating (vowels) versus unvoiced consonants (f, s, k) or silence.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        float: The voiced ratio (0.0 to 1.0).
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    return float(np.sum(voiced_flag) / len(voiced_flag))
