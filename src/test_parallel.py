import numpy as np
from mpi4py import MPI
from parallel_fft import ParallelFFT


def test_parallel_vs_numpy(comm, N):
    """
    Compare ParallelFFT against NumPy FFT for a given N.
    Automatically skips invalid (N, P) combinations.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Four-step FFT requires N divisible by P^2
    if N % (size * size) != 0:
        if rank == 0:
            print(f"[SKIP] N={N} not divisible by P^2={size**2}")
        return True

    # Root creates data
    if rank == 0:
        np.random.seed(0)
        x = np.random.random(N) + 1j * np.random.random(N)
    else:
        x = None

    fft = ParallelFFT(comm)
    y_parallel = fft.parallel_fft(x)

    if rank == 0:
        y_numpy = np.fft.fft(x)
        error = np.max(np.abs(y_parallel - y_numpy))

        if error < 1e-10:
            print(f"[PASS] N={N}, P={size}, error={error:.2e}")
            return True
        else:
            print(f"[FAIL] N={N}, P={size}, error={error:.2e}")
            print("Parallel:", y_parallel)
            print("NumPy:  ", y_numpy)
            return False

    return True


def run_all_tests():
    """
    Run correctness tests across multiple N values.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("PARALLEL FFT  SCALED CORRECTNESS TEST SUITE")
        print(f"Processes: {size}")
        print("=" * 70)

    # Choose sizes that grow gradually
    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    local_results = []

    for N in Ns:
        comm.Barrier()
        result = test_parallel_vs_numpy(comm, N)
        local_results.append(result)

    # Gather results
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = [r for proc in all_results for r in proc]
        print("=" * 70)
        if all(flat_results):
            print("ALL TESTS PASSED ")
        else:
            print("SOME TESTS FAILED ")
        print("=" * 70)
        return all(flat_results)

    return True


if __name__ == "__main__":
    """
    Run with:
      mpiexec -n 1 python -m src.tests.test_parallel
      mpiexec -n 2 python -m src.tests.test_parallel
      mpiexec -n 4 python -m src.tests.test_parallel
    """
    run_all_tests()
