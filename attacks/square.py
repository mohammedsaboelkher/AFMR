"""Audio Watermarking Robustness Benchmarking via Black-Box Square Attack.

This module provides a benchmarking framework to evaluate the adversarial robustness
of state-of-the-art audio watermarking models (AudioSeal, WavMark, and Timbre). 
It implements a query-efficient, black-box decision attack based on the "Square Attack" 
algorithm, adapted specifically for the time-frequency spectrogram representation of audio.

Acoustic Square Attack Mechanics:
1. Representation Space: The 1D audio waveform is mapped into a 2D time-frequency 
   spectrogram using the Short-Time Fourier Transform (STFT). Perturbations are 
   optimized within this 2D space (targeting amplitude, phase, or both) and mapped 
   back via Inverse STFT (iSTFT) to produce the adversarial acoustic wave.
2. Threat Model (Black-Box): The attacker has no access to the target model's 
   gradients, architecture, or weights. It only queries the detector and observes 
   the output token/bit accuracy (the "score").
3. Random Search Optimization: Instead of gradient estimation, the algorithm uses a 
   random search policy. It applies localized patch updates (vertical stripes for 
   $L_{\infty}$ or pseudo-Gaussian tiles for $L_{2}$). If a random update successfully 
   reduces the watermark detection or bit accuracy, the update is accepted; 
   otherwise, it is discarded.
4. Step-Size Scheduling: The size of the localized square patches scales down 
   piece-wise over the iteration space, enabling large structural changes early on 
   and highly localized, imperceptible fine-tuning in later iterations.

Evaluation Metrics Collected:
- Query Count: The number of model evaluations needed to break the watermark.
- Bit Accuracy (Acc): The percentage of the watermark payload correctly recovered.
- Signal-to-Noise Ratio (SNR): Logarithmic ratio of signal to adversarial noise.
- ViSQOL MOS-LQO: Objective metric mimicking human perception of audio quality.
"""

import os
import argparse
import time
from typing import Tuple, Any, List, Union
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import yaml

from audioseal import AudioSeal
import wavmark
from timbre.model.conv2_mel_modules import Decoder
from art.estimators.classification import PyTorchClassifier


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the benchmarking execution.

    Returns:
        argparse.Namespace: Wrapped command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Audio Watermarking Benchmarking with Square Attack")
    
    # Path and Message Arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing your watermarked .wav files")
    parser.add_argument("--secret_message", type=str, required=True, help="The universal bitstring message embedded in all audio files (e.g., 10101010)")
    parser.add_argument("--blackbox_folder", type=str, default="attack_results", help="Folder to save the blackbox attack results")
    
    # Framework Hyperparameters
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5*16000, help="Length of the audio samples")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--save_pert", action="store_true", help="If set, saves the perturbed waveform")
    parser.add_argument("--query_budget", type=int, default=10000, help="Query budget for the attack")
    parser.add_argument("--eps", type=float, default=0.05, help="Epsilon boundary constraint for the attack")
    parser.add_argument("--p", type=float, default=0.05, help="Initial relative size of the coordinate selection patch")
    parser.add_argument("--tau", type=float, default=0.0, help="Threshold bit-accuracy target for the detector (stop condition)")
    parser.add_argument("--snr", type=list, default=[0,2.5, 5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30], help="Signal-to-noise ratio boundaries")
    parser.add_argument("--norm", type=str, default='linf', choices=['linf', 'l2'], help="Norm constraint for the attack")
    parser.add_argument("--attack_bitstring", action="store_true", help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--attack_type", type=str, default='both', choices=['amplitude','phase','both'], help="Spectrogram components to attack")
    parser.add_argument("--model", type=str, default='', choices=['audioseal','wavmark', 'timbre'], required=True, help="Watermarking model architecture to evaluate")

    print("Arguments: ", parser.parse_args())
    return parser.parse_args()
    

def api_visqol() -> Any:
    """Initializes and builds the ViSQOL audio quality evaluation engine API.

    Returns:
        VisqolApi: Ready instance of the Google ViSQOL metric scorer.
    """
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api


np.set_printoptions(precision=5, suppress=True)

def p_selection(p_init: float, it: int, n_iters: int) -> float:
    """Calculates piece-wise scaling constant schedule for coordinates selection.

    Args:
        p_init: The starting ratio fraction.
        it: Current iteration step index.
        n_iters: Max total execution iterations scheduled.

    Returns:
        float: Scaled probability factor controlling patch size.
    """
    it = int(it / n_iters * 10000)
    if 10 < it <= 50: p = p_init / 2
    elif 50 < it <= 200: p = p_init / 4
    elif 200 < it <= 500: p = p_init / 8
    elif 500 < it <= 1000: p = p_init / 16
    elif 1000 < it <= 2000: p = p_init / 32
    elif 2000 < it <= 4000: p = p_init / 64
    elif 4000 < it <= 6000: p = p_init / 128
    elif 6000 < it <= 8000: p = p_init / 256
    elif 8000 < it <= 10000: p = p_init / 512
    else: p = p_init
    return p


def pseudo_gaussian_pert_rectangles(x: int, y: int) -> np.ndarray:
    """Generates a localized rectangular grid filled with pseudo-Gaussian noise weights.

    Args:
        x: Height boundary dimensions of target array space.
        y: Width boundary dimensions of target array space.

    Returns:
        np.ndarray: A normalized 2D matrix of pseudo-Gaussian noise weights.
    """
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1
    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2
        counter2[0] -= 1
        counter2[1] -= 1
    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
    return delta


def meta_pseudo_gaussian_pert(s: int) -> np.ndarray:
    """Structures meta compound boundaries of adversarial shapes using tiles.

    Args:
        s: Length dimensions of the square configuration.

    Returns:
        np.ndarray: Assembled structural multi-tiled perturbation pattern.
    """
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)
    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
    return delta


def square_attack_linf(
    model: PyTorchClassifier, x: np.ndarray, eps: float, n_iters: int, p_init: float, args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    """Executes the Linf bounded configuration of the localized Square Attack.

    Args:
        model: Standardized classifier interface wrapped around the target detector.
        x: Input native clean or initial spectrogram shape matrix.
        eps: Geometric step constraint boundary size.
        n_iters: The query optimization iteration budget limit allocation.
        p_init: Relative localization probability initializer window element.
        args: Captured execution context parameter configurations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tracked array queries counted alongside optimized matrix.
    """
    np.random.seed(0)
    min_val, max_val = x.min(), x.max()
    c, h, w = x.shape[1:]
    n_features = c * h * w
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)
    loss = model.get_detection_result(x_best)
    n_queries = np.ones(x.shape[0])
    time_start = time.time()
    progress_bar = tqdm(range(n_iters), desc='Linf square attack', leave=False)
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0)
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]
        deltas = x_best_curr - x_curr
        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])
        x_new = np.clip(x_curr + deltas, min_val, max_val)
        loss = model.get_detection_result(x_new)
        idx_improved = loss < loss_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1
        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start
        curr_norms_image_best = np.max(np.abs(x_best - x))
        progress_bar.set_description(f'iter: {i_iter+1}, acc: {acc:.2f}, max_pert_best: {curr_norms_image_best:.2f}')
        if acc <= args.tau:
            break
    return n_queries, x_best


def square_attack_l2(
    model: PyTorchClassifier, x: np.ndarray, eps: float, n_iters: int, p_init: float, args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    """Executes the L2 bounded spherical variant of the black-box Square Attack.

    Args:
        model: Standardized classifier interface wrapped around the target detector.
        x: Input native clean or initial spectrogram shape matrix.
        eps: Geometric step constraint boundary size.
        n_iters: The query optimization iteration budget limit allocation.
        p_init: Relative localization probability initializer window element.
        args: Captured execution context parameter configurations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tracked array queries counted alongside optimized matrix.
    """
    np.random.seed(0)
    min_val, max_val = x.min(), x.max()
    c, h, w = x.shape[1:]
    n_features = c * h * w
    delta_init = np.zeros(x.shape)
    s = h // 5
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s
    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, min_val, max_val)
    loss = model.get_detection_result(x_best)
    n_queries = np.ones(x.shape[0])
    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    progress_bar = tqdm(range(n_iters), leave=False)
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0.0)
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]
        delta_curr = x_best_curr - x_curr
        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)
        if s % 2 == 0: s += 1
        s2 = s + 0
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        curr_norms_window = np.sqrt(np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))
        new_deltas = np.ones([x_curr.shape[0], c, s, s]) * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)
        loss = model.get_detection_result(x_new)
        idx_improved = loss < loss_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1
        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start
        curr_norms_image_best = np.sqrt(np.sum((x_best - x) ** 2))
        progress_bar.set_description(f'iter: {i_iter+1}, acc: {acc:.2f}, max_pert_best: {curr_norms_image_best:.2f}')
        if acc <= args.tau:
            break
    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
    return n_queries, x_best


class WatermarkDetectorWrapper(PyTorchClassifier):
    """Unified wrapper standardizing APIs across diverse watermark models."""

    def __init__(
        self, model: torch.nn.Module, message: torch.Tensor, detector_type: str, 
        on_bitstring: bool, transform: Any, th: float, input_size: Tuple[int, ...], 
        model_type: str, device: torch.device
    ) -> None:
        """Initializes the standardized PyTorch classification interface wrapper."""
        super(WatermarkDetectorWrapper, self).__init__(
            model=model, input_shape=input_size, nb_classes=2, channels_first=True, loss=None
        )
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        self.transform = transform
        self.model.to(self._device)
        
        if model_type == 'timbre':
            self.get_detection_result = self.get_detection_result_timbre
            self.bwacc = self.bwacc_timbre
        elif model_type == 'wavmark':
            self.get_detection_result = self.get_detection_result_wavmark
            self.bwacc = self.bwacc_wavmark
        elif model_type == 'audioseal':
            self.get_detection_result = self.get_detection_result_audioseal
            self.bwacc = self.bwacc_audioseal

    def get_detection_result_audioseal(self, spectrogram: np.ndarray) -> np.ndarray:
        """Processes and decodes adversarial configurations using AudioSeal components."""
        spectrogram_tensor = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram_tensor)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None: return np.array([0.0])
            bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
            return np.array([bitacc.item()])
        return np.array([result])

    def bwacc_audioseal(self, signal: torch.Tensor) -> np.ndarray:
        """Returns direct signal evaluation array scores from AudioSeal."""
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None: return np.array([0.0])
            bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
            return np.array([bitacc.item()])
        return np.array([result])

    def get_detection_result_wavmark(self, spectrogram: np.ndarray) -> np.ndarray:
        """Processes and decodes adversarial configurations using WavMark components."""
        spectrogram_tensor = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram_tensor).squeeze(0).detach().cpu()
        payload, _ = wavmark.decode_watermark(self.model, signal)
        if payload is None: return np.array([0.0])
        payload_tensor = torch.tensor(payload).to(self.message.device)
        bit_acc = 1 - torch.sum(torch.abs(payload_tensor - self.message)) / self.message.shape[0]
        return np.array([bit_acc.item()])
        
    def bwacc_wavmark(self, signal: torch.Tensor) -> np.ndarray:
        """Returns direct signal evaluation array scores from WavMark."""
        signal_cpu = signal.squeeze(0).detach().cpu()
        payload, _ = wavmark.decode_watermark(self.model, signal_cpu)
        if payload is None: return np.array([0.0])
        payload_tensor = torch.tensor(payload).to(self.message.device)
        bit_acc = 1 - torch.sum(torch.abs(payload_tensor - self.message)) / self.message.shape[0]
        return np.array([bit_acc.item()])

    def get_detection_result_timbre(self, spectrogram: np.ndarray, batch_size: int = 1) -> np.ndarray:
        """Processes and decodes adversarial configurations using Timbre modules."""
        spectrogram_tensor = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram_tensor)
        payload = self.model.test_forward(signal.unsqueeze(0))
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bitacc.item()])

    def bwacc_timbre(self, signal: torch.Tensor) -> np.ndarray:
        """Returns direct signal evaluation array scores from Timbre."""
        payload = self.model.test_forward(signal.unsqueeze(0))
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bit_acc.item()])
    

class signal22spectrogram:
    """Manages conversion interfaces linking 1D signals to 2D Time-Frequency spaces."""

    def __init__(self, signal: torch.Tensor, low_frequency: int, high_frequency: int, device: torch.device, attack_type: str) -> None:
        """Initializes Fourier transforms and maps spectrum dimensions boundaries."""
        self.signal = signal
        self.attack_type = attack_type
        self.sig2spec = T.Spectrogram(n_fft=400, power=None).to(device)
        self.spec2sig = T.InverseSpectrogram(n_fft=400).to(device)
        self.spectrogram = self.sig2spec(signal)
        self.amplitude = torch.abs(self.spectrogram)
        self.phase = torch.angle(self.spectrogram)
        self.lf = low_frequency
        self.hf = high_frequency
        self.attack_shape = self.spectrogram[..., low_frequency:high_frequency, :].shape
        self.length = signal.shape[-1]

    def signal2spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        """Extracts specified feature target masks from raw domain waveforms."""
        spectro_complex = self.sig2spec(signal)
        spectro_amplitude = torch.abs(spectro_complex)
        spectro_phase = torch.angle(spectro_complex)
        if self.attack_type == 'amplitude': return spectro_amplitude[..., self.lf:self.hf, :]
        elif self.attack_type == 'phase': return spectro_phase[..., self.lf:self.hf, :]
        elif self.attack_type == 'both': return torch.cat([spectro_amplitude, spectro_phase], dim=0)[..., self.lf:self.hf, :]

    def spectrogram2signal(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Synthesizes structured frequency maps back into time waveforms."""
        spectrogram = spectrogram.squeeze()
        if self.attack_type == 'both':
            padding_amp = self.amplitude.clone()
            padding_amp[..., self.lf:self.hf, :] = spectrogram[0]
            padding_phase = self.phase.clone()
            padding_phase[..., self.lf:self.hf, :] = spectrogram[1]
            spectro_complex = padding_amp * torch.exp(1j * padding_phase)
        elif self.attack_type == 'amplitude':
            padding_amp = self.amplitude.clone()
            padding_amp[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = padding_amp * torch.exp(1j * self.phase)
        elif self.attack_type == 'phase':
            padding_phase = self.phase.clone()
            padding_phase[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = self.amplitude * torch.exp(1j * padding_phase)
        signal = self.spec2sig(spectro_complex, self.length)
        return signal
    

def decode_audio_files_perturb_blackbox(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    """Batch-processes and benchmarks blackbox attacks across flat audio folder directories.

    Args:
        model: Loaded watermarking neural network detector module under evaluation.
        args: Parsed command line execution context arguments.
        device: Active compute architecture environment (CUDA/CPU).
    """
    attack = square_attack_l2 if args.norm == 2 else square_attack_linf
    
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory '{args.input_dir}' does not exist.")
        
    watermarked_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.wav')]
    if not watermarked_files:
        print(f"No .wav files found in {args.input_dir}")
        return

    watermarked_files = watermarked_files[:args.testset_size]
    progress_bar = tqdm(watermarked_files, desc="Attacking Audio Dataset")
    os.makedirs(args.blackbox_folder, exist_ok=True)   
    visqol = api_visqol()
    
    original_payload = torch.tensor(list(map(int, args.secret_message)), dtype=torch.int)

    filename = os.path.join(args.blackbox_folder, f'square_spectrogram_results.csv')
    log_exists = os.path.exists(filename)
    with open(filename, 'a' if log_exists else 'w') as log:
        if not log_exists:
            log.write('filename, query, final_acc, snr, visqol\n')

    for watermarked_file in progress_bar:
        file_idx = os.path.splitext(watermarked_file)[0]
        waveform, sample_rate = torchaudio.load(os.path.join(args.input_dir, watermarked_file))
        waveform = waveform.to(device=device)

        transform = signal22spectrogram(waveform, 0, 201, device, args.attack_type)
        detector = WatermarkDetectorWrapper(
            model, original_payload, 'single-tailed', args.attack_bitstring, 
            transform, args.tau, model_type=args.model, input_size=transform.attack_shape, device=device
        )
        
        watermarked_spectrogram = transform.signal2spectrogram(waveform).unsqueeze(0).detach().cpu().numpy()
        n_queries, adv_spectrogram = attack(detector, watermarked_spectrogram, args.eps, args.query_budget, args.p, args)
        
        adv_signal = transform.spectrogram2signal(torch.tensor(adv_spectrogram).to(device))
        acc = detector.bwacc(adv_signal).item()
        
        norm_diff = torch.sum((adv_signal - waveform)**2)
        if norm_diff == 0:
            snr = float('inf')
        else:
            snr = (10 * torch.log10(torch.sum(waveform**2) / norm_diff)).item()
            
        visqol_score = visqol.Measure(
            np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), 
            np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)
        ).moslqo
        
        print(f'\nFile: {watermarked_file} | Queries: {int(n_queries.item())} | Final Acc: {acc:.3f} | SNR: {snr:.1f} | ViSQOL: {visqol_score:.3f}')
        
        with open(filename, 'a') as log:
            log.write(f'{watermarked_file}, {int(n_queries.item())}, {acc}, {snr}, {visqol_score}\n')
            
        torchaudio.save(
            os.path.join(args.blackbox_folder, f"adv_{file_idx}.wav"),
            adv_signal.detach().cpu(), sample_rate
        )


def main() -> None:
    """Main lifecycle driver configuration for orchestrating benchmark testing."""
    args = parse_arguments()

    if args.norm == 'l2':
        args.norm = 2
    else:
        args.norm = np.inf
    
    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
    elif args.model == 'timbre':
        process_config = yaml.load(open("timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        detector = Decoder(process_config, model_config, 30, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load('timbre/results/ckpt/pth/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector

    decode_audio_files_perturb_blackbox(model, args, device)


if __name__ == "__main__":
    main()