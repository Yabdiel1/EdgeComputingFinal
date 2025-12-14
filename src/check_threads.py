import numpy as np
from threadpoolctl import threadpool_info

# Create large array
x = np.random.random(2**26) + 1j*np.random.random(2**26)

# Show which thread pools are being used
print(threadpool_info())

# Measure execution time
import time
start = time.time()
np.fft.fft(x)
end = time.time()
print(f"Execution time: {end - start:.6f} s")
