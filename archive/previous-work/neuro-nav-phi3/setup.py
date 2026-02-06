from setuptools import setup, find_packages

setup(
    name="neuro-nav-phi3",
    version="1.0.0",
    description="3D Scene Graph Construction with Phi-3-Vision (Microsoft)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.0.0",
        "timm>=0.9.0",
        "accelerate>=0.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
        "tyro>=0.5.0",
        "open3d>=0.13.0",
    ],
    python_requires=">=3.8",
)

