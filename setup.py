"""
Setup script for CS 224N Final Project: Sentiment Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cs224n-sentiment-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="zengxiao@stanford.edu",
    description="Advanced Sentiment Analysis with Transformer Models for CS 224N",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zengxiao-he/sentiment-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/zengxiao-he/sentiment-analysis/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-train=src.train:main",
            "sentiment-eval=src.evaluate:main",
        ],
    },
) 