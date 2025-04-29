from setuptools import setup, find_packages

setup(
    name="PyFTE",
    version="0.1.0",
    description="Python for Fault Tree Extraction",
    author="Omar Mostafa",
    author_email="omar.mostafa@kit.edu",
    packages=find_packages(),
    install_requires=[
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)