#!/usr/bin/env python3
"""
Main script for running FFT benchmarks
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from mpi4py import MPI
from src.serial_fft import SerialFFT
from src.parallel_fft import ParallelFFT
from src.utils import generate_test_signal, verify_correctness, save_benchmark_results
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run FFT benchmarks")
    parser.add_argument("--size", type=int, default=2**20, help="Input size (power of 2)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--signal-type", default="composite", help="Test signal type")
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Generate test data (only on root)
    if rank == 0:
        print(f"Generating test signal of size {args.size}...")
        signal = generate_test_signal(args.size, args.signal_type)
        signal_complex = signal.astype(np.complex128)
    else:
        signal_complex = None
    
    # Run serial benchmark on root
    if rank == 0:
        print("\n=== Running Serial FFT Benchmark ===")
        serial_time, serial_result = SerialFFT.benchmark(signal_complex, args.iterations)
        print(f"Serial FFT time: {serial_time:.6f} seconds")
    
    # Run parallel benchmark
    comm.Barrier()
    if rank == 0:
        print("\n=== Running Parallel FFT Benchmark ===")
    
    parallel_fft = ParallelFFT(comm)
    parallel_time, parallel_result = parallel_fft.benchmark(
        signal_complex, args.iterations
    )
    
    # Collect and display results on root
    if rank == 0:
        print(f"Parallel FFT time ({comm.size} processes): {parallel_time:.6f} seconds")
        
        if parallel_result is not None:
            # Verify correctness
            is_correct = verify_correctness(serial_result, parallel_result)
            print(f"Results match: {is_correct}")
            
            # Calculate speedup and efficiency
            speedup = serial_time / parallel_time
            efficiency = (speedup / comm.size) * 100
            
            print(f"\n=== Performance Metrics ===")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Parallel Efficiency: {efficiency:.2f}%")
            
            # Save results
            results = {
                "input_size": args.size,
                "processes": comm.size,
                "serial_time": serial_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "correct": is_correct
            }
            
            save_benchmark_results(results, f"benchmark_{args.size}_{comm.size}procs.json")
            
            # Check if goals are met
            if speedup >= 5.0:
                print("✓ Speedup goal (5x) ACHIEVED!")
            else:
                print(f"✗ Speedup goal (5x) not met: {speedup:.2f}x")
            
            if efficiency >= 60.0:
                print("✓ Efficiency goal (60%) ACHIEVED!")
            else:
                print(f"✗ Efficiency goal (60%) not met: {efficiency:.2f}%")

if __name__ == "__main__":
    main()