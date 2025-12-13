from setuptools import setup, find_packages

setup(
    name="parallel_fft",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "mpi4py>=3.1.3",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8",
)