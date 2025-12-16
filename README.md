# Parallel FFT Benchmark (Serial & MPI)

This project implements and benchmarks **serial** and **parallel** Fast Fourier Transform (FFT) algorithms:

- **Serial FFT** using a recursive Cooley–Tukey implementation
- **Parallel FFT** using **MPI (mpi4py)** and the **Four-Step FFT algorithm**
- Correctness validation against `numpy.fft.fft`
- Designed for **strong-scaling performance analysis**

---

##  Requirements

### 1. Python
- Python **3.9+** recommended

### 2. MPI Implementation
An MPI library must be installed on your system:

- **Linux**:
  ```bash
  sudo apt install openmpi-bin openmpi-common libopenmpi-dev
  ```
- **MacOS**:
  ```bash
  brew install open-mpi
  ```
- **Windows**:
  - Install Microsoft MPI
  - Ensure that `mpiexec` is available in your PATH

**Verify MPI Installation**:
  ```bash
  mpiexec -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
  ```
## Python Dependencies

 Install required Python packages
 ```bash
  pip install numpy mpi4py
 ```

## Run Serial FFT Benchmark

The serial FFT benchmark measures execution time for increasing input size with different FFT implementation
- Recursive FFT
- Vectorized FFT
- Numpy FFT

```bash
python src/serial_fft.py
```

## Run Parallel FFT Benchmark

The parallel FFT uses MPI processes (not threads).

From the root directory of the project
```bash
mpiexec -n <P> python src/parallel_fft.py
```
Where P = Number of MPI processes
### Note:
For correctness, parallel fft requires:
  - `N` is a power of 2
  - `N` divisible by P^2, where P is the nunber of MPI processes
  
  
  

