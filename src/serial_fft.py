import numpy as np
import time
from typing import Tuple

class SerialFFT:
    """Serial implementation of FFT using Cooley-Tukey algorithm"""
    
    @staticmethod
    def fft(x: np.ndarray) -> np.ndarray:
        """
        Compute 1D FFT using Cooley-Tukey algorithm
        Assumes length of x is a power of 2
        """
        n = len(x)
        
        if n <= 1:
            return x
        
        # Divide
        even = SerialFFT.fft(x[::2])
        odd = SerialFFT.fft(x[1::2])
        
        # Combine
        t = np.exp(-2j * np.pi * np.arange(n // 2) / n)
        return np.concatenate([even + t * odd, even - t * odd])
    
    @staticmethod
    def ifft(x: np.ndarray) -> np.ndarray:
        """Inverse FFT"""
        n = len(x)
        conj_fft = SerialFFT.fft(np.conjugate(x))
        return np.conjugate(conj_fft) / n
    
    @staticmethod
    def benchmark(x: np.ndarray, iterations: int = 10) -> Tuple[float, np.ndarray]:
        """Benchmark the FFT computation"""
        times = []
        result = None
        
        for _ in range(iterations):
            start = time.perf_counter()
            result = SerialFFT.fft(x)
            end = time.perf_counter()
            times.append(end - start)
        
        return np.mean(times), result