import math
import numpy as np
import time
from typing import Tuple

class SerialFFT:
    """Serial implementation of FFT using Cooley-Tukey algorithm"""

    @staticmethod
    def dft(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)
    
    @staticmethod
    def FFT(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]

        if N & (N - 1) != 0:
            raise ValueError("size of x must be a power of 2")

        if N <= 32:
            return SerialFFT.dft(x)

        X_even = SerialFFT.FFT(x[::2])
        X_odd  = SerialFFT.FFT(x[1::2])

        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.concatenate([
            X_even + factor[:N // 2] * X_odd,
            X_even + factor[N // 2:] * X_odd
        ])

    @staticmethod
    def FFT_vectorized(x):
        """Vectorized, non-recursive Cooley–Tukey FFT"""
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]

        if N & (N - 1) != 0:
            raise ValueError("size of x must be a power of 2")

        N_min = min(N, 32)

        # Initial DFTs
        n = np.arange(N_min)
        k = n[:, None]
        M = np.exp(-2j * np.pi * n * k / N_min)
        X = np.dot(M, x.reshape((N_min, -1)))

        # Iterative buildup
        while X.shape[0] < N:
            half = X.shape[1] // 2
            X_even = X[:, :half]
            X_odd  = X[:, half:]

            factor = np.exp(
                -1j * np.pi * np.arange(X.shape[0]) / X.shape[0]
            )[:, None]

            X = np.vstack([
                X_even + factor * X_odd,
                X_even - factor * X_odd
            ])

        return X.ravel()
    
    
    @staticmethod
    def benchmark(
        x: np.ndarray,
        method: str = "FFT",
        iterations: int = 10
    ) -> Tuple[float, np.ndarray]:
        """
        Benchmark selected FFT method.

        method options:
        - "FFT"            : recursive Cooley–Tukey
        - "FFT_vectorized" : vectorized Cooley–Tukey
        - "numpy"          : np.fft.fft
        """
        methods = {
            "FFT": SerialFFT.FFT,
            "FFT_vectorized": SerialFFT.FFT_vectorized,
            "numpy": np.fft.fft,
        }

        if method not in methods:
            raise ValueError(f"Unknown FFT method: {method}")

        fft_func = methods[method]

        times = []
        result = None

        for _ in range(iterations):
            start = time.perf_counter()
            result = fft_func(x)
            end = time.perf_counter()
            times.append(end - start)

        return np.mean(times), result

    

# methods = ["FFT", "FFT_vectorized", "numpy"]

# for method in methods:
#     print(f"\n--- Benchmarking method: {method} ---")
#     for sizes in [2**i for i in range(4, 25)]:
#         x = np.random.random(sizes) + 1j * np.random.random(sizes)
#         avg_time, _ = SerialFFT.benchmark(x, method=method, iterations=5)
#         print(
#             f"Method: {method:14s} | "
#             f"Size: {sizes:6d} (2^{int(math.log2(sizes))}) | "
#             f"Avg Time: {avg_time:.6f} s"
#         )
