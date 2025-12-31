from setuptools import setup, find_packages

setup(
    name="fluxions",
    version="0.1.0",
    description="A deep learning framework built from scratch",
    packages=find_packages(),  # Automatically finds 'fluxions' and 'fluxions.core'
    install_requires=[
        "numpy<2.0.0",
    ],
    python_requires=">=3.8",
)
