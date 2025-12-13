import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

print(f"Current directory: {os.getcwd()}")
print(f"This file location: {__file__}")

# Solution A: Try different import approaches
try:
    # Option 1: Relative import (requires running as module)
    from ..serial_fft import SerialFFT
    print("✓ Import successful using relative import")
except ImportError as e:
    print(f"Relative import failed: {e}")
    
    # Option 2: Add src to path and import directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Go from tests/ to src/
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    print(f"Trying to import from: {parent_dir}")
    print(f"Files in {parent_dir}: {os.listdir(parent_dir)}")
    
    try:
        from serial_fft import SerialFFT
        print("✓ Import successful using sys.path method")
    except ImportError as e2:
        print(f"Import failed: {e2}")
        print("Available modules in parent directory:")
        for f in os.listdir(parent_dir):
            if f.endswith('.py'):
                print(f"  - {f}")
        raise

def test_basic_fft():
    """Test FFT with simple known cases"""
    print("=" * 50)
    print("Test 1: Basic FFT Tests")
    print("=" * 50)
    
    # Test 1: Single element
    x1 = np.array([1.0])
    result1 = SerialFFT.fft(x1)
    expected1 = np.array([1.0])
    assert np.allclose(result1, expected1), f"Single element test failed: {result1} != {expected1}"
    print("✓ Single element test passed")
    
    # Test 2: Two elements
    x2 = np.array([1.0, 0.0])
    result2 = SerialFFT.fft(x2)
    expected2 = np.array([1.0, 1.0])
    assert np.allclose(result2, expected2), f"Two element test failed: {result2} != {expected2}"
    print("✓ Two element test passed")
    
    # Test 3: Simple cosine wave
    n = 8
    t = np.arange(n)
    x3 = np.cos(2 * np.pi * t / n)
    result3 = SerialFFT.fft(x3)
    numpy_result3 = np.fft.fft(x3)
    assert np.allclose(result3, numpy_result3, rtol=1e-10), "Cosine wave test failed"
    print("✓ Cosine wave test passed")
    
    # Test 4: Random data
    np.random.seed(42)
    x4 = np.random.random(16)
    result4 = SerialFFT.fft(x4)
    numpy_result4 = np.fft.fft(x4)
    assert np.allclose(result4, numpy_result4, rtol=1e-10), "Random data test failed"
    print("✓ Random data test passed")
    
    print("All basic FFT tests passed! ✅")


def test_inverse_fft():
    """Test that IFFT(FFT(x)) ≈ x"""
    print("\n" + "=" * 50)
    print("Test 2: Inverse FFT Tests")
    print("=" * 50)
    
    test_cases = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.cos(np.linspace(0, 2*np.pi, 16)),
        np.random.random(32),
        np.random.random(64) + 1j * np.random.random(64)  # Complex input
    ]
    
    for i, x in enumerate(test_cases):
        # Forward FFT
        x_fft = SerialFFT.fft(x)
        # Inverse FFT
        x_reconstructed = SerialFFT.ifft(x_fft)
        
        # Check if we get back the original (within tolerance)
        assert np.allclose(x, x_reconstructed, rtol=1e-10), \
            f"Inverse FFT test {i+1} failed. Max error: {np.max(np.abs(x - x_reconstructed))}"
        print(f"✓ Inverse FFT test {i+1} passed (size: {len(x)})")
    
    print("All inverse FFT tests passed! ✅")


def test_power_of_two():
    """Test that FFT works for various power-of-two sizes"""
    print("\n" + "=" * 50)
    print("Test 3: Power-of-Two Size Tests")
    print("=" * 50)
    
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    for size in sizes:
        # Generate test data
        x = np.random.random(size) + 1j * np.random.random(size)
        
        # Compute FFT using our implementation
        our_result = SerialFFT.fft(x)
        
        # Compute FFT using NumPy for comparison
        numpy_result = np.fft.fft(x)
        
        # Compare results
        error = np.max(np.abs(our_result - numpy_result))
        assert error < 1e-10, f"Size {size} test failed with error {error}"
        print(f"✓ Size {size:4d}: max error = {error:.2e}")
    
    print("All power-of-two tests passed! ✅")


def benchmark_performance():
    """Benchmark performance against NumPy's FFT"""
    print("\n" + "=" * 50)
    print("Test 4: Performance Benchmark")
    print("=" * 50)
    
    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    our_times = []
    numpy_times = []
    
    print(f"{'Size':>8} | {'Our FFT (ms)':>12} | {'NumPy FFT (ms)':>12} | {'Ratio':>8}")
    print("-" * 55)
    
    for size in sizes:
        # Generate test data
        x = np.random.random(size) + 1j * np.random.random(size)
        
        # Benchmark our implementation
        iterations = 5 if size <= 512 else 3
        our_time, _ = SerialFFT.benchmark(x, iterations=iterations)
        our_times.append(our_time * 1000)  # Convert to milliseconds
        
        # Benchmark NumPy's implementation
        numpy_start = time.perf_counter()
        for _ in range(iterations):
            _ = np.fft.fft(x)
        numpy_time = (time.perf_counter() - numpy_start) / iterations * 1000
        numpy_times.append(numpy_time)
        
        ratio = our_time * 1000 / numpy_time if numpy_time > 0 else float('inf')
        print(f"{size:8d} | {our_time*1000:12.4f} | {numpy_time:12.4f} | {ratio:8.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, our_times, 'b-o', label='Our FFT', linewidth=2)
    plt.plot(sizes, numpy_times, 'r--s', label='NumPy FFT', linewidth=2)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Input Size (log scale, base 2)')
    plt.ylabel('Time (ms, log scale)')
    plt.title('FFT Performance Comparison')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fft_performance.png', dpi=150)
    print("\nPerformance plot saved as 'fft_performance.png'")
    
    print("\nNote: NumPy's FFT is highly optimized (uses FFTW library)")
    print("Our implementation is for educational purposes")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "=" * 50)
    print("Test 5: Edge Cases")
    print("=" * 50)
    
    # Test with complex zeros
    x1 = np.zeros(8, dtype=complex)
    result1 = SerialFFT.fft(x1)
    assert np.allclose(result1, np.zeros(8)), "All zeros test failed"
    print("✓ All zeros test passed")
    
    # Test with real zeros
    x2 = np.zeros(16, dtype=float)
    result2 = SerialFFT.fft(x2)
    assert np.allclose(result2, np.zeros(16)), "Real zeros test failed"
    print("✓ Real zeros test passed")
    
    # Test with DC signal (all ones)
    x3 = np.ones(32)
    result3 = SerialFFT.fft(x3)
    numpy_result3 = np.fft.fft(x3)
    assert np.allclose(result3, numpy_result3, rtol=1e-10), "DC signal test failed"
    print("✓ DC signal test passed")
    
    # Test with impulse signal
    x4 = np.zeros(64)
    x4[0] = 1.0
    result4 = SerialFFT.fft(x4)
    numpy_result4 = np.fft.fft(x4)
    assert np.allclose(result4, numpy_result4, rtol=1e-10), "Impulse signal test failed"
    print("✓ Impulse signal test passed")
    
    print("All edge case tests passed! ✅")


def test_parseval():
    """Test Parseval's theorem: sum(|x|²) = sum(|X|²)/N"""
    print("\n" + "=" * 50)
    print("Test 6: Parseval's Theorem")
    print("=" * 50)
    
    test_sizes = [8, 16, 32, 64, 128]
    
    for size in test_sizes:
        # Generate random signal
        x = np.random.random(size) + 1j * np.random.random(size)
        
        # Compute FFT
        X = SerialFFT.fft(x)
        
        # Parseval's theorem: sum(|x|²) = sum(|X|²)/N
        energy_time = np.sum(np.abs(x) ** 2)
        energy_freq = np.sum(np.abs(X) ** 2) / size
        
        error = np.abs(energy_time - energy_freq)
        assert error < 1e-10, f"Parseval's theorem failed for size {size}"
        print(f"✓ Size {size:4d}: |Σ|x|² - Σ|X|²/N| = {error:.2e}")
    
    print("Parseval's theorem verified for all test cases! ✅")


def run_all_tests():
    """Run all tests"""
    print("Starting SerialFFT Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_basic_fft()
        test_inverse_fft()
        test_power_of_two()
        test_edge_cases()
        test_parseval()
        benchmark_performance()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print(f"Total test time: {total_time:.2f} seconds")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 60)
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()