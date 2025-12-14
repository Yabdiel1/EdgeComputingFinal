import math
import numpy as np
from mpi4py import MPI
from threadpoolctl import threadpool_limits
from parallel_fft import ParallelFFT
from serial_fft import SerialFFT

def strong_scaling_benchmark(N_list, iterations=3, debug=False, use_naive=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    results = []

    for N in N_list:
        # Only root generates input
        if rank == 0:
            x = np.random.random(N) + 1j*np.random.random(N)
        else:
            x = None

        # --- Serial FFT baseline ---
        if rank == 0:
            if use_naive:
                # Naive recursive FFT
                serial_times = []
                for _ in range(iterations):
                    start = MPI.Wtime()
                    SerialFFT.FFT(x)
                    end = MPI.Wtime()
                    serial_times.append(end - start)
            else:
                # NumPy FFT with single thread
                serial_times = []
                with threadpool_limits(limits=1):
                    for _ in range(iterations):
                        start = MPI.Wtime()
                        np.fft.fft(x)
                        end = MPI.Wtime()
                        serial_times.append(end - start)
            serial_time = np.mean(serial_times)
        else:
            serial_time = None

        # Broadcast serial time
        serial_time = comm.bcast(serial_time, root=0)

        # --- Parallel FFT ---
        parallel_fft = ParallelFFT(comm, debug=debug)
        parallel_times = []
        for _ in range(iterations):
            comm.Barrier()
            start = MPI.Wtime()
            parallel_fft.parallel_fft(x)
            comm.Barrier()
            end = MPI.Wtime()
            parallel_times.append(end - start)

        parallel_time = np.mean(parallel_times)

        # --- Speedup and efficiency ---
        speedup = serial_time / parallel_time if parallel_time > 0 else 0.0
        efficiency = speedup / size * 100

        if rank == 0:
            print("="*70)
            print(f"STRONG SCALING BENCHMARK: N={N} (2^({int(math.log2(N))})), P={size}")
            print(f"Serial Time   : {serial_time:.6f} s")
            print(f"Parallel Time : {parallel_time:.6f} s")
            print(f"Speedup       : {speedup:.2f}x")
            print(f"Efficiency    : {efficiency:.1f}%")
            print("="*70)
            results.append((N, size, serial_time, parallel_time, speedup, efficiency))

    if rank == 0:
        return results
    return None


if __name__ == "__main__":
    # Example: test multiple input sizes
    N_list = [2**i for i in range(10, 27)]  # You can adjust based on memory
    # Set use_naive=True to benchmark against the naive SerialFFT instead of NumPy
    strong_scaling_benchmark(N_list, iterations=3, debug=False, use_naive=True)



