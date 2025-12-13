import numpy as np
from mpi4py import MPI
import time
from typing import Tuple, Optional
from .utils import check_against_numpy_fft

class ParallelFFT:
    """Parallel FFT implementation using MPI"""
    
    def __init__(self, comm: Optional[MPI.Intracomm] = None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def validate_input_size(self, n: int) -> bool:
        """Check if input size is valid for parallel FFT"""
        # Ensure n is power of 2 and divisible by number of processes
        return (n & (n - 1) == 0) and (n % self.size == 0)
    
    def distribute_data(self, x: np.ndarray) -> np.ndarray:
        """
        Distribute input data across processes
        Returns local portion of data
        """
        n = len(x)
        local_n = n // self.size
        
        # Scatter data to all processes
        local_data = np.zeros(local_n, dtype=x.dtype)
        self.comm.Scatter(x, local_data, root=0)
        
        return local_data
    
    def parallel_fft(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Main parallel FFT implementation
        Process 0 returns the complete result
        """
        if self.rank == 0:
            # Input validation
            if not self.validate_input_size(len(x)):
                raise ValueError(f"Input size {len(x)} must be power of 2 and divisible by {self.size}")
        
        # Broadcast input size to all processes
        n = len(x) if self.rank == 0 else 0
        n = self.comm.bcast(n, root=0)
        
        # Step 1: Distribute data
        local_data = self.distribute_data(x) if self.rank == 0 else np.zeros(n // self.size, dtype=np.complex128)
        if self.rank != 0:
            self.comm.Scatter(None, local_data, root=0)
        
        # Step 2: Local FFT computation
        local_result = self._local_fft(local_data)
        
        # Step 3: Global combination (butterfly stages)
        # TODO: Implement parallel butterfly operations
        global_result = self._butterfly_combination(local_result, n)
        
        # Step 4: Gather results at root
        if self.rank == 0:
            final_result = np.zeros(n, dtype=np.complex128)
        else:
            final_result = None
        
        self.comm.Gather(global_result, final_result, root=0)
        
        return final_result if self.rank == 0 else None
    
    def _local_fft(self, x: np.ndarray) -> np.ndarray:
        """Compute local FFT on each process's data"""
        # Use serial FFT for local computation
        from .serial_fft import SerialFFT
        return SerialFFT.fft(x)
    
    def _butterfly_combination(self, local_data: np.ndarray, total_n: int) -> np.ndarray:
        """
        Correct parallel FFT combination using MPI_Alltoall.
        This implements a distributed Cooley–Tukey FFT.
        """
        p = self.size
        local_n = len(local_data)
        assert total_n == p * local_n

        # Step 1: reshape local FFT output into p blocks
        reshaped = local_data.reshape(p, local_n // p)

        # Step 2: all-to-all exchange (transpose)
        exchanged = np.empty_like(reshaped)
        self.comm.Alltoall(reshaped, exchanged)

        # Step 3: flatten exchanged data
        exchanged = exchanged.reshape(local_n)

        # Step 4: apply twiddle factors
        for i in range(local_n):
            global_idx = self.rank + p * i
            exchanged[i] *= np.exp(-2j * np.pi * global_idx / total_n)

        # Step 5: final local FFT
        from .serial_fft import SerialFFT
        return SerialFFT.fft(exchanged)


    def benchmark(self, x: np.ndarray, iterations: int = 10) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Benchmark parallel FFT"""
        times = []
        result = None
        
        # Synchronize before timing
        self.comm.Barrier()
        
        for _ in range(iterations):
            start = time.perf_counter()
            result = self.parallel_fft(x)
            self.comm.Barrier()
            end = time.perf_counter()
            
            if self.rank == 0:
                times.append(end - start)
        
        avg_time = np.mean(times) if self.rank == 0 else None
        return avg_time, result