import numpy as np
from mpi4py import MPI
from .serial_fft import SerialFFT
from .parallel_fft import ParallelFFT

def test_parallel_vs_serial_small():
    """Compare parallel FFT with serial FFT for small input (N=8)"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 8
    if N % size != 0:
        if rank == 0:
            print(f"Skipping test: N={N} not divisible by size={size}")
        return True

    # Generate test data on root ONLY
    if rank == 0:
        np.random.seed(42)
        x = np.random.random(N) + 1j * np.random.random(N)
    else:
        x = None  # Non-root processes don't need the data

    # Create ParallelFFT instance and run
    parallel_fft = ParallelFFT(comm)
    parallel_result = parallel_fft.parallel_fft(x)

    if rank == 0:
        serial_result = SerialFFT.fft(x)
        error = np.max(np.abs(parallel_result - serial_result))
        if error < 1e-10:
            print(f"[PASS] Parallel vs Serial small FFT test (N={N})")
        else:
            print(f"[FAIL] Parallel vs Serial small FFT test (N={N}) - Max error: {error:.2e}")
            # Debug output
            print(f"Parallel result: {parallel_result}")
            print(f"Serial result: {serial_result}")
        return error < 1e-10

    return True

def test_data_distribution():
    """Test if distribute_data correctly splits the array"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 8
    if N % size != 0:
        if rank == 0:
            print(f"Skipping data distribution test: N={N} not divisible by size={size}")
        return True

    # Root creates sequential data
    if rank == 0:
        x = np.arange(N, dtype=np.complex128)
    else:
        x = None

    # Create ParallelFFT instance
    parallel_fft = ParallelFFT(comm)
    local_data = parallel_fft.distribute_data(x) if rank == 0 else np.zeros(N // size, dtype=np.complex128)
    if rank != 0:
        parallel_fft.comm.Scatter(None, local_data, root=0)

    # Verify each process has correct chunk
    start = rank * (N // size)
    end = start + (N // size)
    expected = np.arange(start, end, dtype=np.complex128)
    if np.array_equal(local_data, expected):
        print(f"Process {rank}: Data distribution correct")
        return True
    else:
        print(f"Process {rank}: Data distribution incorrect")
        print(f"  Expected: {expected}")
        print(f"  Got: {local_data}")
        return False

def run_all_tests():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*50)
        print("Parallel FFT Minimal Test Suite")
        print(f"Running with {comm.Get_size()} processes")
        print("="*50)

    results = []
    results.append(test_data_distribution())
    comm.Barrier()
    results.append(test_parallel_vs_serial_small())
    comm.Barrier()

    all_results = comm.gather(results, root=0)
    if rank == 0:
        flat_results = [r for proc in all_results for r in proc]
        if all(flat_results):
            print("ALL MINIMAL TESTS PASSED")
        else:
            print("SOME MINIMAL TESTS FAILED")
        return all(flat_results)
    return True

if __name__ == "__main__":
    # Run with MPI
    # e.g., mpiexec -n 2 python -m src.tests.parallel_test
    run_all_tests()
