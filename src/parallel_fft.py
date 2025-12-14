import numpy as np
from mpi4py import MPI
from typing import Optional, Tuple

class ParallelFFT:
    """
    Parallel FFT implementation using MPI (Transpose-Split / Four-Step Method).
    Requires N to be divisible by P^2.
    """

    def __init__(self, comm: Optional[MPI.Intracomm] = None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def validate_input_size(self, n: int) -> bool:
        """
        Check if input size is valid for this parallel implementation.
        N must be a power of 2 and divisible by P^2 for clean block distribution.
        """
        return (n & (n - 1) == 0) and (n % (self.size * self.size) == 0)

    def distribute_data(self, x: np.ndarray) -> np.ndarray:
        """Scatter input data across processes."""
        n = len(x)
        local_n = n // self.size
        local_data = np.zeros(local_n, dtype=np.complex128)
        # Use DOUBLE_COMPLEX for np.complex128
        self.comm.Scatter([x, MPI.DOUBLE_COMPLEX], [local_data, MPI.DOUBLE_COMPLEX], root=0)
        return local_data

    def parallel_fft(self, x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Parallel 1D FFT using the Four-Step algorithm.
        
        Algorithm for N-point FFT with P processes (requires N = P * L where L = P * M):
        Input index: n = i * L + j where i in [0,P), j in [0,L)
        Output index: k = k1 * L + k2 where k1 in [0,P), k2 in [0,L)
        
        X[k] = X[k1*L + k2] = sum_{i=0}^{P-1} sum_{j=0}^{L-1} x[i*L+j] * W_N^{(i*L+j)*(k1*L+k2)}
        
        Four steps:
        1. Local FFT over j: Y[i,k2] = sum_j x[i,j] * W_L^{j*k2}
        2. Twiddle: Z[i,k2] = Y[i,k2] * W_N^{i*k2}
        3. Transpose: Z'[k2,i] = Z[i,k2]
        4. Local FFT over i: X[k2,k1] = sum_i Z'[k2,i] * W_P^{i*k1}
        Output: X[k1*L + k2]
        """
        # --- Setup ---
        if self.rank == 0:
            if x is None:
                raise ValueError("Root process must provide input array")
            n = len(x)
            if not self.validate_input_size(n):
                raise ValueError(f"Input size {n} must be power of 2 and divisible by P^2={self.size**2}")
        else:
            n = None

        n = self.comm.bcast(n, root=0)
        P = self.size
        L = n // P
        
        # --- Distribute: Each process gets row i (elements [i*L, ..., i*L+L-1]) ---
        local_data = np.zeros(L, dtype=np.complex128)
        if self.rank == 0:
            self.comm.Scatter([x, MPI.DOUBLE_COMPLEX], [local_data, MPI.DOUBLE_COMPLEX], root=0)
        else:
            self.comm.Scatter(None, [local_data, MPI.DOUBLE_COMPLEX], root=0)

        # --- Step 1: Local FFT of length L ---
        # Process rank holds x[rank*L : rank*L+L]
        # Compute Y[rank, k2] = FFT_L(x[rank, :])
        Y = np.fft.fft(local_data)
        
        # --- Step 2: Apply twiddle factors W_N^{i*k2} ---
        # i = self.rank, k2 = 0, 1, ..., L-1
        i = self.rank
        k2_indices = np.arange(L)
        twiddles = np.exp(-2j * np.pi * i * k2_indices / n)
        Z = Y * twiddles
        
        # --- Step 3: Transpose (All-to-All) ---
        # We need to transpose from (i, k2) to (k2, i) layout
        # Process rank currently holds Z[rank, 0:L]
        # After transpose, process rank should hold Z[rank*block : (rank+1)*block, 0:P]
        # where block = L // P
        
        block = L // P
        if L % P != 0:
            raise ValueError(f"L={L} must be divisible by P={P}")
        
        # Reshape: Z is currently a 1D array of length L
        # We want to send chunk [k*block : (k+1)*block] to process k
        sendbuf = Z.reshape(P, block).copy()  # Shape: (P, block)
        recvbuf = np.empty((P, block), dtype=np.complex128)
        
        self.comm.Alltoall([sendbuf, MPI.DOUBLE_COMPLEX], [recvbuf, MPI.DOUBLE_COMPLEX])
        
        # After Alltoall:
        # recvbuf[i, m] came from process i, positions [rank*block + m]
        # This is Z[i, rank*block + m] in the original (i, k2) indexing
        # Which represents the element for row i, column (rank*block + m)
        
        # --- Step 4: Local FFT of length P ---
        # We now have P values for each of our "columns" (values of k2 in our range)
        # recvbuf has shape (P, block) where axis-0 is the i index
        # We need FFT along axis-0
        X_local = np.fft.fft(recvbuf, axis=0)  # Shape: (P, block)
        
        # X_local[k1, m] is the FFT output for:
        #   k1 in [0, P), m in [0, block)
        #   This corresponds to output index k = k1 * L + (rank * block + m)
        
        # --- Step 5: Gather ---
        if self.rank == 0:
            gather_buf = np.empty((P, P, block), dtype=np.complex128)
        else:
            gather_buf = None
        
        self.comm.Gather([X_local, MPI.DOUBLE_COMPLEX], [gather_buf, MPI.DOUBLE_COMPLEX], root=0)
        
        if self.rank == 0:
            # gather_buf[src_rank, k1, m] = X[k1, src_rank * block + m]
            # This is the output for index k = k1 * L + src_rank * block + m
            result = np.empty(n, dtype=np.complex128)
            for src_rank in range(P):
                for k1 in range(P):
                    for m in range(block):
                        k = k1 * L + src_rank * block + m
                        result[k] = gather_buf[src_rank, k1, m]
            return result
        
        return None


    def benchmark(self, x: np.ndarray, iterations: int = 5) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Benchmark parallel FFT."""
        times = []
        result = None
        self.comm.Barrier()

        for _ in range(iterations):
            start = MPI.Wtime()
            result = self.parallel_fft(x)
            self.comm.Barrier()
            end = MPI.Wtime()
            if self.rank == 0:
                times.append(end - start)

        avg_time = np.mean(times) if self.rank == 0 else None
        return avg_time, result
