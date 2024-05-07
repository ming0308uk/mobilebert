from setuptools import setup, find_packages

setup(
    name="mobile_bert",
    version="0.1.0",
    description="A Python package for working with MobileBERT",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/ming0308uk/mobilebert",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
