from setuptools import setup, find_packages

setup(
    name="signal_processing_project",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "plotly>=5.3.0",
        "scipy>=1.7.0",
        "pywavelets>=1.1.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "opencv-python>=4.5.0",
        "streamlit>=1.10.0",
        "pillow>=8.0.0",
    ],
    author="Sameer M",
    author_email="sameerm1421999@gmail.com",
    description="Wave Feature Extraction for Medical Imaging Applications",
    keywords="signal processing, medical imaging, wave analysis, feature extraction",
    url="https://github.com/SamSon1402/signal-processing-project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "wave-feature-extractor=app:main",
        ],
    },
)