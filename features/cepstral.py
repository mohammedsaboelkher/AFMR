import librosa
import numpy as np


def get_mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict[str, np.ndarray]:
    """Calculates statistical summaries of Mel-Frequency Cepstral Coefficients.

    MFCCs mathematically represent the "shape" of the human vocal tract.
    Deltas (velocity) measure how the shape changes, and Delta-Deltas (acceleration)
    measure how fast that change is changing. Together, they fingerprint articulation.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.

    Returns:
        dict[str, np.ndarray]: A dictionary containing 1D arrays for 'mfcc_mean',
            'mfcc_var', 'delta_mean', and 'delta2_mean'.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return {
        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_var": np.var(mfcc, axis=1),
        "delta_mean": np.mean(delta_mfcc, axis=1),
        "delta2_mean": np.mean(delta2_mfcc, axis=1),
    }
