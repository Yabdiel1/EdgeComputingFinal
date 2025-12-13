import time
import numpy as np
import sys
import os
from mpi4py import MPI
from .serial_fft import SerialFFT
from .parallel_fft import ParallelFFT
from .utils import check_against_numpy_fft

def test_parallel_vs_serial_small():
    """Compare parallel FFT with serial FFT for small inputs"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only test sizes divisible by number of processes
    n = 16  # Small size for testing
    if n % size != 0:
        if rank == 0:
            print(f"Skipping test: {n} not divisible by {size}")
        return
    
    # Generate test data
    if rank == 0:
        np.random.seed(42)
        x = np.random.random(n) + 1j * np.random.random(n)
    else:
        x = None
    
    # Broadcast data to all processes
    x = comm.bcast(x, root=0)
    
    # Create parallel FFT instance
    parallel_fft = ParallelFFT(comm)
    
    # Compute parallel FFT
    parallel_result = parallel_fft.parallel_fft(x)
    
    # Compare with serial FFT (only on root process)
    if rank == 0:
        serial_result = SerialFFT.fft(x)
        
        # Check if results match
        error = np.max(np.abs(parallel_result - serial_result))
        
        if error < 1e-10:
            print(f"Parallel vs Serial test passed for n={n}, size={size}")
            print(f"  Max error: {error:.2e}")
        else:
            print(f"Parallel vs Serial test failed for n={n}, size={size}")
            print(f"  Max error: {error:.2e}")
            
        return error < 1e-10
    return True

def test_validation():
    """Test input validation"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    parallel_fft = ParallelFFT(comm)
    
    if rank == 0:
        # Test invalid size (not power of 2)
        invalid_n = 15
        x_invalid = np.random.random(invalid_n)
        
        try:
            _ = parallel_fft.parallel_fft(x_invalid)
            print("Validation test failed: Should have raised error for non-power-of-2")
            return False
        except ValueError as e:
            print(f"Correctly raised error for non-power-of-2: {e}")
        
        # Test size not divisible by number of processes
        size = comm.Get_size()
        invalid_n2 = 32
        if invalid_n2 % size != 0:
            x_invalid2 = np.random.random(invalid_n2)
            try:
                _ = parallel_fft.parallel_fft(x_invalid2)
                print("Validation test failed: Should have raised error for non-divisible size")
                return False
            except ValueError as e:
                print(f"Correctly raised error for non-divisible size: {e}")
    
    comm.Barrier()
    return True

def test_distribution():
    """Test data distribution across processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = 32
    if n % size != 0:
        if rank == 0:
            print(f"Skipping distribution test: {n} not divisible by {size}")
        return
    
    parallel_fft = ParallelFFT(comm)
    
    if rank == 0:
        # Create sequential data for easy verification
        x = np.arange(n, dtype=np.complex128)
    else:
        x = None
    
    # Test distribute_data method
    local_n = n // size
    expected_data = None
    
    if rank == 0:
        # Manually calculate what each process should receive
        for i in range(size):
            start = i * local_n
            end = (i + 1) * local_n
            if i == 0:
                expected_data = x[start:end]
            # Send expected data to each process
            if i != 0:
                comm.send(x[start:end], dest=i, tag=1)
    else:
        expected_data = comm.recv(source=0, tag=1)
    
    # Now use the actual distribution method
    if rank == 0:
        local_data = parallel_fft.distribute_data(x)
    else:
        local_data = np.zeros(local_n, dtype=np.complex128)
        parallel_fft.comm.Scatter(None, local_data, root=0)
    
    # Verify each process got the right data
    if np.array_equal(local_data, expected_data):
        print(f"Process {rank}: Data distribution correct")
    else:
        print(f"Process {rank}: Data distribution incorrect")
        print(f"  Expected: {expected_data}")
        print(f"  Got: {local_data}")
        return False
    
    return True

def test_benchmark():
    """Benchmark parallel FFT performance"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Use a larger size for meaningful benchmarking
    n = 1024
    if n % size != 0:
        if rank == 0:
            print(f"Skipping benchmark: {n} not divisible by {size}")
        return
    
    parallel_fft = ParallelFFT(comm)
    
    if rank == 0:
        np.random.seed(42)
        x = np.random.random(n) + 1j * np.random.random(n)
    else:
        x = None
    
    x = comm.bcast(x, root=0)
    
    # Benchmark
    iterations = 3  # Fewer iterations for MPI tests
    avg_time, result = parallel_fft.benchmark(x, iterations=iterations)
    
    if rank == 0:
        # Also benchmark serial for comparison
        serial_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = SerialFFT.fft(x)
            end = time.perf_counter()
            serial_times.append(end - start)
        
        serial_avg = np.mean(serial_times)
        
        print(f"\nBenchmark Results (n={n}, processes={size}):")
        print(f"  Parallel FFT: {avg_time:.6f} seconds")
        print(f"  Serial FFT:   {serial_avg:.6f} seconds")
        print(f"  Speedup:      {serial_avg/avg_time:.2f}x")
        
        # Verify results are correct
        serial_result = SerialFFT.fft(x)
        error = np.max(np.abs(result - serial_result))
        if error < 1e-10:
            print(f"  Results verified (max error: {error:.2e})")
        else:
            print(f"  Results incorrect (max error: {error:.2e})")
    
    return True
def test_parallel_vs_numpy_multi_size():
        """
        Compare parallel FFT against NumPy FFT for multiple sizes.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sizes = [256, 512, 1024, 2048, 4096]

        all_passed = True

        for n in sizes:
            if n % size != 0:
                if rank == 0:
                    print(f"[SKIP] n={n} not divisible by {size}")
                continue

            if rank == 0:
                np.random.seed(123)
                x = np.random.random(n)
            else:
                x = None

            x = comm.bcast(x, root=0)

            parallel_fft = ParallelFFT(comm)
            result = parallel_fft.parallel_fft(x)

            if rank == 0:
                ok = check_against_numpy_fft(x, result)
                status = "PASS" if ok else "FAIL"
                print(f"[{status}] n={n}")

                all_passed &= ok

        return all_passed

def run_all_tests():
    """Run all parallel FFT tests"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("Parallel FFT Test Suite")
        print(f"Running with {comm.Get_size()} processes")
        print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Validation
    if rank == 0:
        print("\n1. Testing input validation...")
    comm.Barrier()
    test_results.append(test_validation())
    
    # Test 2: Data distribution
    if rank == 0:
        print("\n2. Testing data distribution...")
    comm.Barrier()
    test_results.append(test_distribution())
    
    # Test 3: Parallel vs Serial comparison
    if rank == 0:
        print("\n3. Testing parallel vs serial FFT...")
    comm.Barrier()
    test_results.append(test_parallel_vs_serial_small())
    
    # Test 4: Benchmark
    if rank == 0:
        print("\n4. Benchmarking performance...")
    comm.Barrier()
    test_results.append(test_benchmark())
    
    # Collect results
    all_results = comm.gather(test_results, root=0)
    
    if rank == 0:
        # Flatten results
        flat_results = []
        for proc_results in all_results:
            flat_results.extend(proc_results)
        
        # Check if all tests passed
        all_passed = all(flat_results)
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL PARALLEL TESTS PASSED!")
        else:
            print("SOME TESTS FAILED")
        print("=" * 60)
        
        return all_passed
    return True

if __name__ == "__main__":
    # Note: This must be run with MPI
    # e.g., mpiexec -n 4 python -m src.tests.parallel_test
    run_all_tests()