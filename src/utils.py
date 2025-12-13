import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import json
import os

def generate_test_signal(n: int, signal_type: str = 'composite') -> np.ndarray:
    """Generate test signals for FFT"""
    t = np.linspace(0, 1, n, endpoint=False)
    
    if signal_type == 'composite':
        # Composite signal with multiple frequencies
        signal = (np.sin(2 * np.pi * 50 * t) + 
                  0.5 * np.sin(2 * np.pi * 120 * t) +
                  0.3 * np.sin(2 * np.pi * 300 * t))
    elif signal_type == 'impulse':
        # Impulse signal
        signal = np.zeros(n)
        signal[n // 2] = 1
    elif signal_type == 'random':
        # Random signal
        signal = np.random.randn(n)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal

def verify_correctness(serial_result: np.ndarray, 
                      parallel_result: np.ndarray,
                      tolerance: float = 1e-10) -> bool:
    """Verify that serial and parallel results match"""
    return np.allclose(serial_result, parallel_result, rtol=tolerance)

def plot_results(original: np.ndarray, 
                 transformed: np.ndarray,
                 title: str = "FFT Results"):
    """Plot original and transformed signals"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain - original
    axes[0, 0].plot(original.real if np.iscomplexobj(original) else original)
    axes[0, 0].set_title("Original Signal (Time Domain)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    
    # Frequency domain - magnitude
    n = len(transformed)
    freq = np.fft.fftfreq(n)
    magnitude = np.abs(transformed)
    
    axes[0, 1].plot(freq[:n//2], magnitude[:n//2])
    axes[0, 1].set_title("FFT Magnitude Spectrum")
    axes[0, 1].set_xlabel("Frequency")
    axes[0, 1].set_ylabel("Magnitude")
    
    # Frequency domain - phase
    phase = np.angle(transformed)
    axes[1, 0].plot(freq[:n//2], phase[:n//2])
    axes[1, 0].set_title("FFT Phase Spectrum")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Phase (radians)")
    
    # Time domain - reconstructed (inverse)
    axes[1, 1].plot(np.fft.ifft(transformed).real)
    axes[1, 1].set_title("Reconstructed Signal (IFFT)")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Amplitude")
    
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.show()
    
def check_against_numpy_fft(x, parallel_result, rtol=1e-10, atol=1e-12):
    """
    Compare parallel FFT result against NumPy FFT on root.
    """
    reference = np.fft.fft(x)
    ok = np.allclose(parallel_result, reference, rtol=rtol, atol=atol)

    if not ok:
        diff = np.max(np.abs(parallel_result - reference))
        print(f"[ERROR] FFT mismatch: max |diff| = {diff:.3e}")

    return ok

def save_benchmark_results(results: dict, filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON file"""
    os.makedirs("results/benchmarks", exist_ok=True)
    
    with open(f"results/benchmarks/{filename}", "w") as f:
        json.dump(results, f, indent=2)