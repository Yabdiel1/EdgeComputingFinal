import numpy as np
from mpi4py import MPI
from typing import Optional, Tuple

class ParallelFFT:
    """
    Parallel FFT implementation using MPI (Four-Step Method with CORRECT indexing).
    Requires N to be divisible by P^2.
    
    KEY FIX: Uses COLUMN-MAJOR indexing (n = m*P + p) instead of row-major!
    """

    def __init__(self, comm: Optional[MPI.Intracomm] = None, debug: bool = False):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.debug = debug

    def _debug_print(self, message: str, force_all: bool = False):
        """Print debug message with rank prefix"""
        if self.debug:
            if force_all or self.rank == 0:
                print(f"[Rank {self.rank}] {message}")
                
    def _debug_barrier(self, message: str = ""):
        """Synchronize and print message from rank 0"""
        if self.debug:
            self.comm.Barrier()
            if self.rank == 0 and message:
                print(f"\n{'='*70}")
                print(message)
                print('='*70)
            self.comm.Barrier()

    def validate_input_size(self, n: int) -> bool:
        """
        Check if input size is valid for this parallel implementation.
        N must be a power of 2 and divisible by P^2 for clean block distribution.
        """
        return (n & (n - 1) == 0) and (n % (self.size * self.size) == 0)

    def parallel_fft(self, x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Parallel 1D FFT using the Four-Step algorithm with CORRECT indexing.
        
        CRITICAL: Input is reshaped in COLUMN-MAJOR order (Fortran-style)
        This means: n = m*P + p (not p*M + m)
        
        Algorithm:
        1. Reshape input as P×M matrix in COLUMN-MAJOR order
        2. Each process gets one row (M elements)
        3. Local FFT of length M on each row
        4. Apply twiddle factors W_N^(p*k2)
        5. Transpose via Alltoall
        6. Local FFT of length P on each column
        7. Gather and reorder using k = k1*M + k2
        """
        # --- Setup ---
        if self.rank == 0:
            if x is None:
                raise ValueError("Root process must provide input array")
            n = len(x)
            if not self.validate_input_size(n):
                raise ValueError(
                    f"Input size {n} must be power of 2 and divisible by P^2={self.size**2}"
                )
            if self.debug:
                print(f"\n{'='*70}")
                print(f"PARALLEL FFT (CORRECTED - COLUMN-MAJOR)")
                print(f"{'='*70}")
                print(f"Input: {x}")
                if n <= 16:
                    print(f"Expected: {np.fft.fft(x)}")
        else:
            n = None

        n = self.comm.bcast(n, root=0)
        P = self.size
        M = n // P
        block = M // P
        
        if M % P != 0:
            raise ValueError(f"M={M} must be divisible by P={P}")
        
        self._debug_barrier(f"CONFIGURATION: N={n}, P={P}, M={M}, block={block}")
        
        # --- Step 1: Reshape in COLUMN-MAJOR order and distribute ---
        self._debug_barrier("STEP 1: Reshape (COLUMN-MAJOR) and distribute")
        
        if self.rank == 0:
            # CRITICAL: Reshape in Fortran (column-major) order!
            # This means matrix[p, m] = x[m*P + p]
            matrix = x.reshape((P, M), order='F')
            if self.debug and n <= 16:
                print("Matrix (column-major):")
                print(matrix)
                print("\nIndex mapping verification:")
                for p in range(min(P, 2)):
                    for m in range(min(M, 4)):
                        n_idx = m * P + p
                        print(f"  x[{n_idx}] = matrix[{p},{m}] = {matrix[p,m]}")
        else:
            matrix = None
        
        # Scatter rows
        local_data = np.zeros(M, dtype=np.complex128)

        if self.rank == 0:
            sendbuf = np.empty((P, M), dtype=np.complex128)
            for p in range(P):
                sendbuf[p, :] = matrix[p, :]
        else:
            sendbuf = None

        self.comm.Scatter(
            [sendbuf, MPI.DOUBLE_COMPLEX],
            [local_data, MPI.DOUBLE_COMPLEX],
            root=0
        )

        
        self._debug_print(f"Row {self.rank}: {local_data}", force_all=True)
        
        # --- Step 2: Local FFT of length M ---
        self._debug_barrier("STEP 2: Local Row FFTs (length M)")
        
        Y = np.fft.fft(local_data)
        
        self._debug_print(f"After FFT: {Y}", force_all=True)
        
        # --- Step 3: Apply twiddle factors W_N^(p*k2) ---
        self._debug_barrier("STEP 3: Apply Twiddle Factors W_N^(p*k2)")
        
        p = self.rank
        k2_indices = np.arange(M)
        twiddles = np.exp(-2j * np.pi * p * k2_indices / n)
        Z = Y * twiddles
        
        if self.debug and M <= 8:
            self._debug_print(f"Twiddle factors (p={p}): {twiddles}", force_all=True)
        self._debug_print(f"After twiddle: {Z}", force_all=True)
        
        # --- Step 4: Transpose (Alltoall) ---
        self._debug_barrier("STEP 4: Transpose via Alltoall")
        
        sendbuf = Z.reshape(P, block).copy()
        recvbuf = np.empty((P, block), dtype=np.complex128)
        
        if self.debug and block <= 4:
            self._debug_print(f"Send buffer:\n{sendbuf}", force_all=True)
        
        self.comm.Alltoall([sendbuf, MPI.DOUBLE_COMPLEX], [recvbuf, MPI.DOUBLE_COMPLEX])
        
        if self.debug and block <= 4:
            self._debug_print(f"Receive buffer:\n{recvbuf}", force_all=True)
        
        # --- Step 5: Local FFT of length P ---
        self._debug_barrier("STEP 5: Local Column FFTs (length P)")
        
        X_local = np.fft.fft(recvbuf, axis=0)
        
        self._debug_print(f"After column FFT:\n{X_local}", force_all=True)
        
        # --- Step 6: Gather and Reorder ---
        self._debug_barrier("STEP 6: Gather and Reorder")
        
        if self.rank == 0:
            gather_buf = np.empty((P, P, block), dtype=np.complex128)
        else:
            gather_buf = None
        
        self.comm.Gather([X_local, MPI.DOUBLE_COMPLEX], [gather_buf, MPI.DOUBLE_COMPLEX], root=0)
        
        if self.rank == 0:
            if self.debug and block <= 2:
                print(f"Gather buffer shape: {gather_buf.shape}")
                for src in range(P):
                    print(f"From Process {src}:\n{gather_buf[src]}")
            
            # Reorder: k = k1*M + k2 where k2 = src_rank*block + m
            result = np.empty(n, dtype=np.complex128)
            
            if self.debug and n <= 16:
                print(f"\nReordering: k = k1*M + k2")
                print(f"{'Proc':<6} {'k1':<4} {'k2':<4} {'k':<4} {'Value':<30}")
                print('-'*60)
            
            for src_rank in range(P):
                for k1 in range(P):
                    for m in range(block):
                        k2 = src_rank * block + m
                        k = k1 * M + k2
                        result[k] = gather_buf[src_rank, k1, m]
                        
                        if self.debug and n <= 16:
                            val = gather_buf[src_rank, k1, m]
                            val_str = f"{val.real:.4f}+{val.imag:.4f}i" if val.imag >= 0 else f"{val.real:.4f}{val.imag:.4f}i"
                            print(f"{src_rank:<6} {k1:<4} {k2:<4} {k:<4} {val_str:<30}")
            
            if self.debug:
                print(f"\n{'='*70}")
                print("FINAL RESULT:")
                print('='*70)
                print(f"Result: {result}")
                if n <= 16:
                    expected = np.fft.fft(x)
                    print(f"Expected: {expected}")
                    error = np.max(np.abs(result - expected))
                    print(f"\nMax error: {error:.2e}")
                    if error < 1e-10:
                        print(" RESULT MATCHES NUMPY FFT! ")
                    else:
                        print(" RESULT DOES NOT MATCH")
                        for i in range(n):
                            match = "[MATCH]" if np.abs(result[i] - expected[i]) < 1e-10 else "[MISMATCH]"
                            print(f"  [{i}] Result: {result[i]:.6f}, Expected: {expected[i]:.6f} {match}")
                print('='*70)
            
            return result
        
        return None

    def benchmark(self, x: np.ndarray, iterations: int = 5):
        """
        Benchmark parallel FFT.
        Returns average parallel runtime (max across ranks).
        """
        times = []
        self.comm.Barrier()

        for _ in range(iterations):
            self.comm.Barrier()
            start = MPI.Wtime()
            _ = self.parallel_fft(x)
            self.comm.Barrier()
            end = MPI.Wtime()

            elapsed = end - start
            max_elapsed = self.comm.reduce(elapsed, op=MPI.MAX, root=0)

            if self.rank == 0:
                times.append(max_elapsed)

        avg_time = np.mean(times) if self.rank == 0 else None
        return avg_time



# Test script
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N_sizes = [2**i for i in range(10, 27)]

    fft = ParallelFFT(comm, debug=False)

    for N in N_sizes:
        if rank == 0:
            print("\n" + "="*70)
            print(f"TESTING PARALLEL FFT: N={N}, P={size}")
            print("="*70)
            x = np.random.random(N) + 1j * np.random.random(N)
        else:
            x = None

        # --- Timing ---
        comm.Barrier()
        start = MPI.Wtime()

        y_parallel = fft.parallel_fft(x)

        comm.Barrier()
        end = MPI.Wtime()

        elapsed = end - start
        max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)

        # --- Correctness check ---
        if rank == 0:
            y_numpy = np.fft.fft(x)
            error = np.max(np.abs(y_parallel - y_numpy))

            print(f"Parallel Time : {max_time:.6f} s")
            print(f"Max Error     : {error:.2e}")

            if error < 1e-10:
                print(" Test passed!")
            else:
                print(" Test failed!")
