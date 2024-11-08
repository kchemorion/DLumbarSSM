from setuptools import setup, find_packages

setup(
    name="spine_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.4",
        "pydicom>=2.2.2",
        "opencv-python>=4.5.3",
        "scikit-image>=0.18.3",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.4.3",
        "scipy>=1.7.1",
        "tqdm>=4.62.0",
        "seaborn>=0.11.2"
    ],
    python_requires=">=3.8",
)