import os
import librosa
import numpy as np
import pandas as pd

from typing import Any

from basic import (
    get_duration,
    get_sample_rate,
    get_peak_amplitude,
    get_rms_energy,
    get_dynamic_range,
    get_silence_ratio,
    get_zero_crossing_rate,
    get_crest_factor,
)

from spectral import (
    get_spectral_centroid,
    get_spectral_bandwidth,
    get_spectral_rolloff,
    get_spectral_flatness,
    get_spectral_entropy,
    get_spectral_flux,
    get_spectral_contrast,
)

from harmonic import (
    get_f0_mean_variance,
    get_jitter,
    get_shimmer,
    get_hnr,
    get_voiced_ratio,
)

from cepstral import get_mfcc_stats

from temporal import (
    get_speech_rate,
    get_pause_statistics,
    get_modulation_spectrum,
    get_energy_envelope_variance,
)

from perceptual import (
    get_lufs,
    get_loudness,
    get_sharpness,
    get_roughness,
)

from distortion import (
    get_snr,
    get_log_spectral_distance,
    get_mel_cepstral_distortion,
    get_stft_difference,
    get_pesq_score,
    get_stoi_score,
    get_total_phase_error,
)


def extract_pre_watermark_features(audio_path: str) -> dict[str, Any]:
    """Orchestrates Layers 1-6 for a single audio file.

    Loads the audio file into memory once and distributes the array to the
    feature extraction functions. Flattens array-based features for pandas.

    Args:
        audio_path (str): The absolute or relative path to the audio file.

    Returns:
        dict[str, Any]: A flat dictionary of all extracted features.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    features: dict[str, Any] = {
        "file_name": os.path.basename(audio_path),
        "file_path": audio_path,
    }

    features["duration"] = get_duration(y, sr)
    features["sample_rate"] = get_sample_rate(sr)
    features["peak_amplitude"] = get_peak_amplitude(y)
    features["rms_energy"] = get_rms_energy(y)
    features["dynamic_range"] = get_dynamic_range(y)
    features["silence_ratio"] = get_silence_ratio(y)
    features["zero_crossing_rate"] = get_zero_crossing_rate(y)
    features["crest_factor"] = get_crest_factor(y)

    features["spectral_centroid"] = get_spectral_centroid(y, sr)
    features["spectral_bandwidth"] = get_spectral_bandwidth(y, sr)
    features["spectral_rolloff"] = get_spectral_rolloff(y, sr)
    features["spectral_flatness"] = get_spectral_flatness(y)
    features["spectral_entropy"] = get_spectral_entropy(y)
    features["spectral_flux"] = get_spectral_flux(y)
    features["spectral_contrast"] = get_spectral_contrast(y, sr)

    f0_mean, f0_var = get_f0_mean_variance(audio_path)
    features["f0_mean"] = f0_mean
    features["f0_variance"] = f0_var
    features["jitter"] = get_jitter(audio_path)
    features["shimmer"] = get_shimmer(audio_path)
    features["hnr"] = get_hnr(audio_path)
    features["voiced_ratio"] = get_voiced_ratio(y, sr)

    mfcc_stats = get_mfcc_stats(y, sr, n_mfcc=13)
    for stat_name, array_vals in mfcc_stats.items():
        for i, val in enumerate(array_vals):
            features[f"{stat_name}_{i + 1}"] = float(val)

    features["speech_rate"] = get_speech_rate(y, sr)
    pause_mean, pause_var = get_pause_statistics(y, sr)
    features["pause_mean"] = pause_mean
    features["pause_variance"] = pause_var
    features["modulation_spectrum"] = get_modulation_spectrum(y, sr)
    features["energy_envelope_variance"] = get_energy_envelope_variance(y)

    features["lufs"] = get_lufs(y, sr)
    features["loudness"] = get_loudness(y)
    features["sharpness"] = get_sharpness(y, sr)
    features["roughness"] = get_roughness(y, sr)

    return features


def extract_watermark_distortion(orig_path: str, wm_path: str) -> dict[str, Any]:
    """Orchestrates Layer 7 to measure audio degradation.

    Loads both files, ensures their arrays are precisely the same length
    (to prevent mathematical crashes), and calculates distortion metrics.

    Args:
        orig_path (str): Path to the clean, original audio.
        wm_path (str): Path to the watermarked/distorted audio.

    Returns:
        dict[str, Any]: A flat dictionary of distortion metrics.
    """
    features: dict[str, Any] = {
        "orig_file": os.path.basename(orig_path),
        "wm_file": os.path.basename(wm_path),
    }

    y_orig, sr_orig = librosa.load(orig_path, sr=None, mono=True)
    y_wm, sr_wm = librosa.load(wm_path, sr=None, mono=True)

    if sr_orig != sr_wm:
        y_wm = librosa.resample(y_wm, orig_sr=sr_wm, target_sr=sr_orig)
        sr_wm = sr_orig

    target_length = min(len(y_orig), len(y_wm))
    y_orig = librosa.util.fix_length(y_orig, size=target_length)
    y_wm = librosa.util.fix_length(y_wm, size=target_length)

    features["snr"] = get_snr(y_orig, y_wm)
    features["log_spectral_distance"] = get_log_spectral_distance(y_orig, y_wm)
    features["mel_cepstral_distortion"] = get_mel_cepstral_distortion(
        y_orig, y_wm, sr_orig
    )
    features["stft_difference"] = get_stft_difference(y_orig, y_wm)
    features["pesq_score"] = get_pesq_score(y_orig, y_wm, sr_orig)
    features["stoi_score"] = get_stoi_score(y_orig, y_wm, sr_orig)
    features["total_phase_error"] = get_total_phase_error(y_orig, y_wm)

    return features


def orchestrate_audio_features(
    audio_path: str, wm_path: str | None = None, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Orchestrates the entire feature extraction process for a single audio file.

    Args:
        audio_path (str): Path to the original audio file.
        wm_path (str | None): Optional path to the watermarked/distorted audio file.
        metadata (dict[str, Any] | None): Optional metadata to include in the features.

    Returns:
        dict[str, Any]: A dictionary containing all extracted features.
    """
    pre_watermark_features = extract_pre_watermark_features(audio_path)

    if wm_path:
        distortion_features = extract_watermark_distortion(audio_path, wm_path)
        combined_features = {**pre_watermark_features, **distortion_features}
    else:
        combined_features = pre_watermark_features

    if metadata:
        combined_features.update(metadata)

    return combined_features
